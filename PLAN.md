# PLAN.md — Sketches Under the Microscope: How Tight Are Count-Min's Guarantees on Skewed Data?

**Course:** CS328 Introduction to Data Science (Solo Project, 10 hours total budget)

**Time Budget (10.0 hrs total):** Data + cleaning + Zipf: `1.0h` | `sketch.py` + core tests + human audit: `1.5h` | Harness + CI utilities: `1.0h` | Exp (a): `0.75h` | Exp (b) (reduced): `1.0h` | $d=1$ ablation: `0.5h` | Plots (2 figures): `1.0h` | Report writing & interpretation (by student): `2.5h` | Buffer: `0.75h`

**Target Environment:** Agentic Coding IDE (Google Antigravity)

**Execution Mode:** Strict specification — make **zero** autonomous analysis, modeling, or parameter decisions.

## 1. Project Goal and Research Questions

This project empirically evaluates how tight the worst-case Markov-based additive error guarantee of the Count-Min Sketch (CMS) is when applied to skewed word-frequency distributions (both a real four-novel Fyodor Dostoevsky corpus of $N \approx 10^6$ tokens and synthetic Zipfian streams with $s \in \{0.6, 0.8, 1.0, 1.2, 1.5\}$). A Count-Min Sketch configured with width $w = \lceil e / \epsilon \rceil$ and depth $d = \lceil \ln(1 / \delta) \rceil$ guarantees that for any fixed item $x$ with true count $c(x)$, its estimated count $\hat{c}(x)$ satisfies $\Pr[\hat{c}(x) - c(x) > \epsilon N] \le \delta$. Because this bound relies on Markov's inequality in each row and ignores heavy-hitter concentration, it may be very loose in practice, while the additive error $\epsilon N$ simultaneously yields large relative errors on rare items. We answer two concrete questions:

* **Question (a) — Empirical Failure Probability vs. True Frequency:** How does the empirical failure probability of the Count-Min guarantee ($\hat{c}(x) - c(x) > \epsilon N$) vary with an item's true frequency $c(x)$? Specifically, how often does the guarantee fail across $T$ independent hash-function draws for representative items from the **head**, **middle**, and **tail** of the frequency distribution compared to the theoretical ceiling $\delta$, and how much of the error reduction comes from taking the minimum across $d$ rows versus a single row ($d = 1$)?

* **Question (b) — Relative-Error Frequency Threshold vs. Skew:** What minimum true frequency $f^*$ is required so that at least $95\%$ of independent trials achieve a relative error $\frac{\hat{c}(x) - c(x)}{c(x)} \le 0.10$, and how does this threshold $f^*$ change as the Zipf skew exponent $s$ increases from $0.6$ to $1.5$?

## 2. Repository Layout

The repository must strictly follow this structure (no extra packages, nested modules, or configuration frameworks):

```
.
├── PLAN.md                  # This specification document
├── requirements.txt         # Pinned dependencies (numpy, scipy, matplotlib, pandas, pytest)
├── data.py                  # Gutenberg downloader, header/footer stripper, tokenizer, and Zipf generator
├── sketch.py                # From-scratch Carter-Wegman hash family and linear bincount Count-Min Sketch
├── experiments.py           # Seeding, trial harness, Clopper-Pearson CIs, Exp (a), Exp (b), and d=1 ablation
├── plots.py                 # Matplotlib generation of Figure 1 and Figure 2 PNGs
├── demo.ipynb               # Optional minimal 3-cell static notebook calling run_demo(w=907, s=1.0)
├── data/
│   ├── raw/                 # Cached raw Project Gutenberg .txt files (gitignored)
│   └── processed/           # Saved .npz / .csv summary outputs from experiments.py
├── figures/                 # Output directory for Figure 1 and Figure 2 PNGs
└── tests/
    ├── test_data.py         # Verification of mock-string header stripping, regex tokenization, N and V range
    ├── test_sketch.py       # Tiny hand-checkable case (V=5, w=3, d=2), one-sided error, d=1 equivalence
    └── test_experiments.py  # Per-item Clopper-Pearson CI checks and NaN-threshold test
```

## 3. Locked Decisions ("Do Not Deviate")

The agent **must not** deviate from any of the following ten requirements:

 1. **From-Scratch NumPy Implementation:** Implement all hashing and Count-Min Sketch operations from scratch using `numpy`. External sketching libraries (e.g., `probables` or `countminsketch`) may only be imported inside `tests/` for secondary cross-checking, never in `sketch.py` or `experiments.py`.

 2. **Carter-Wegman Universal Hash Family:** Each row $r \in \{0, \dots, d-1\}$ uses an independent hash function:

    $$
    h_r(x) = ((a_r \cdot x + b_r) \bmod p) \bmod w
    $$

    where $p = 2^{31} - 1 = 2147483647$ (Mersenne prime), $a_r \in \{1, \dots, p-1\}$ is drawn uniformly at random, and $b_r \in \{0, \dots, p-1\}$ is drawn uniformly at random. Parameters $(a_r, b_r)$ are drawn independently per row and redrawn independently per trial. All arithmetic must use `np.int64` so that $a_r \cdot x + b_r$ never overflows (since $p \approx 2.15 \times 10^9$ and $x < V \ll p$, $a_r \cdot x + b_r < p^2 \approx 4.61 \times 10^{18} < 2^{63} - 1 \approx 9.22 \times 10^{18}$).

 3. **Linearity Exploit (No Token-by-Token Streaming in Trials):** Because the Count-Min Sketch is a linear sketch, the final counter table depends only on the exact item frequency vector $c = [c(0), c(1), \dots, c(V-1)]$. **Never** iterate over the $N \approx 10^6$ token stream inside a trial loop. Instead, compute `exact_counts` once per dataset/stream, and construct row $r$ via:

    ```
    buckets_r = np.bincount(h_r(ids), weights=exact_counts, minlength=w).astype(np.int64)
    ```

    Point queries for an array of item IDs `q_ids` are computed as:

    $$
    \hat{c}(x) = \min_{r \in \{0, \dots, d-1\}} \text{table}[r, h_r(x)]
    $$

 4. **Failure Definition for Question (a):** For a fixed item $x$, trial $t \in \{1, \dots, T\}$ is a failure if and only if:

    $$
    \hat{c}^{(t)}(x) - c(x) > \epsilon N
    $$

    where $N = \sum_{x=0}^{V-1} c(x)$ is the total token count of the stream and $\epsilon$ is the target error parameter used to set $w = \lceil e / \epsilon \rceil$.

 5. **Per-Item Clopper-Pearson Confidence Intervals:** Compute $95\%$ Clopper-Pearson exact binomial confidence intervals **per item** across the $T$ independent trials using:

    ```
    scipy.stats.binomtest(k, n=T).proportion_ci(confidence_level=0.95, method="exact")
    ```

    where $k$ is the number of failures for item $x$ across $T$ trials. **Never pool failures across distinct items**, because items within the same trial share the same random hash functions $(a_r, b_r)$ and are not independent.

 6. **Representative Items by Frequency Rank:** Select exactly $20$ representative items per frequency class (**head**, **middle**, **tail**) based on 1-based frequency rank $r \in \{1, \dots, V\}$ sorted descending by exact count $c(x)$ (ties broken deterministically by ascending integer ID):

    * **Head ($20$ items):** Ranks $r \in \{1, 2, \dots, 20\}$.

    * **Middle ($20$ items, evaluated at primary $\epsilon = 0.003$ only):** $20$ evenly spaced ranks centered around the median rank $\lfloor V / 2 \rfloor$: specifically `np.linspace(V // 2 - 95, V // 2 + 95, 20, dtype=int)`.

    * **Tail ($20$ items):** $20$ evenly spaced ranks in the bottom decile $[0.90 V, V]$ among items with $c(x) > 0$: specifically `np.linspace(int(0.90 * V_pos), V_pos, 20, dtype=int)` where $V_{\text{pos}}$ is the number of items with $c(x) > 0$.

    * *Scope Rule:* At the primary $\epsilon = 0.003$, evaluate all three classes (`head`, `middle`, `tail`). At secondary $\epsilon \in \{0.01, 0.001\}$, evaluate `head` and `tail` only.

 7. **Threshold Definition for Question (b):** For a given stream and sketch configuration (run at primary $\epsilon = 0.003, d = 3$), compute for every item $x$ with $c(x) > 0$ the empirical success rate across $T$ trials:

    $$
    q(x) = \frac{1}{T} \sum_{t=1}^T \mathbb{I}\!\left[\frac{\hat{c}^{(t)}(x) - c(x)}{c(x)} \le 0.10\right]
    $$

    Compute the primary global binned threshold `f_binned_global` ($f_{\text{binned}}^*$) using $30$ logarithmically spaced frequency bins `np.logspace(0, np.log10(c_max), 31)` (smallest bin lower edge where all higher-or-equal non-empty bins have mean success rate $\ge 0.95$). Also compute `f_strict_global` ($f_{\text{strict}}^*$, the smallest observed frequency $f$ such that every observed frequency $f' \ge f$ has mean pass rate $\bar{q}(f') \ge 0.95$, which is trivial to compute alongside `f_binned_global`). Do **not** compute regime-specific head/tail frequency thresholds. If no frequency or bin satisfies the $95\%$ criterion, record the threshold as `NaN`. Do not substitute any other value.

 8. **Relative Error Restriction:** Relative error $\frac{\hat{c}(x) - c(x)}{c(x)}$ is only defined and computed for items with $c(x) > 0$. Any synthetic Zipf item with $c(x) = 0$ must be excluded from relative-error calculations.

 9. **Reproducible Seeding:** Use `np.random.default_rng(seed)` with a deterministic, isolated seed per experimental configuration (defined in Section 4.2) so that running any experiment in isolation produces bit-for-bit identical results.

10. **Parameter Sweep Structure:** Because Markov's bound is conservative, small $\epsilon$ may yield zero empirical failures ($k = 0$). Use $(\epsilon = 0.003, \delta = 0.05) \implies (w = 907, d = 3)$ as the **primary configuration** everywhere. In Experiment (a) and the $d = 1$ ablation, sweep $\epsilon \in \{0.01, 0.003, 0.001\}$ (with all 3 classes at $\epsilon = 0.003$ and `head`/`tail` only at $\epsilon \in \{0.01, 0.001\}$). In Experiment (b), run **only** at the primary $\epsilon = 0.003$ across the Zipf skew values $s \in \{0.6, 0.8, 1.0, 1.2, 1.5\}$ plus the real Dostoevsky corpus.

## 4. Explicit Parameter Grid

The agent must hardcode the following exact constants in `experiments.py` and `data.py`:

### 4.1 Corpus and Synthetic Stream Parameters

* **Real Corpus (4 Dostoevsky texts, Constance Garnett translation, UTF-8 plain text from Project Gutenberg):**

  1. *The Brothers Karamazov* — Gutenberg eBook ID `28054` (`https://www.gutenberg.org/cache/epub/28054/pg28054.txt`)

  2. *Crime and Punishment* — Gutenberg eBook ID `2554` (`https://www.gutenberg.org/cache/epub/2554/pg2554.txt`)

  3. *Notes from the Underground* — Gutenberg eBook ID `600` (`https://www.gutenberg.org/cache/epub/600/pg600.txt`)

  4. *The Possessed* (The Devils) — Gutenberg eBook ID `8117` (`https://www.gutenberg.org/cache/epub/8117/pg8117.txt`)

* **Header/Footer Stripping Rule:** For each file, locate the first line starting with `*** START OF` (case-insensitive) and the last line starting with `*** END OF` (case-insensitive). Keep strictly the lines between these two markers (excluding the marker lines themselves). Concatenate the four cleaned texts, convert to lowercase, and extract tokens matching regex `[a-z']+`.

* **ID Mapping:** Sort unique words by descending frequency (ties broken alphabetically by word string) and assign integer IDs $0, 1, \dots, V-1$ so that ID $x$ corresponds to 1-based rank $r = x + 1$.

* **Synthetic Zipf Streams:** Using the exact $N$ and $V$ measured from the real Dostoevsky corpus, generate synthetic count vectors for each skew exponent:

  $$
  s \in \{0.6, 0.8, 1.0, 1.2, 1.5\}
  $$

  For a given $s$, define rank probabilities $p_r = \frac{r^{-s}}{\sum_{j=1}^V j^{-s}}$ for $r \in \{1, \dots, V\}$. Draw a single fixed synthetic count vector of total length $N$ via `rng_zipf.multinomial(N, p)` using seed `1000 + int(s * 100)`, re-sorting counts descending so rank order is preserved. (Do not generate or check a separate deterministic rounded vector.)

### 4.2 Sketch and Trial Parameters

* **Mersenne Prime:** $p = 2^{31} - 1 = 2147483647$.

* **Error Parameter Sweep ($\epsilon$):**

  * $\epsilon_1 = 0.01 \implies w_1 = \lceil e / 0.01 \rceil = 272$ (Exp (a) and ablation only; `head` and `tail` classes only)

  * $\epsilon_2 = 0.003 \implies w_2 = \lceil e / 0.003 \rceil = 907$ **(Primary width — used in Exp (a), Exp (b), and ablation; all 3 classes)**

  * $\epsilon_3 = 0.001 \implies w_3 = \lceil e / 0.001 \rceil = 2719$ (Exp (a) and ablation only; `head` and `tail` classes only)

* **Failure Probability / Depth ($\delta, d$):**

  * Primary setting: $\delta = 0.05 \implies d = \lceil \ln(1 / 0.05) \rceil = \lceil 2.99573 \rceil = 3$.

  * Ablation setting: $d = 1$ (single row, where theoretical Markov failure bound is $\delta_{d=1} = 1 / e \approx 0.3679$).

* **Number of Independent Hash Trials ($T$) and Batching:**

  * During development and smoke-testing, use $T_{\text{dev}} = 500$; for all final saved runs, use $T = 5000$ independent trials (provides exact Clopper-Pearson upper bound of $\approx 0.00074$ when $k = 0$, well below $\delta = 0.05$).

  * Trials must be processed in chunks of at most $250$ trials per batch (`BATCH_SIZE = 250`), so that no array of shape $(T, V)$ is ever materialized for the full $T = 5000$ (a $(5000 \times V)$ `int64` array is about $1\text{ GB}$ per intermediate).

* **Head vs. Tail Split for Question (b) / Figure 2:**

  * Define the **Head regime** as ranks $r \le 500$ (top $500$ most frequent items) and the **Tail regime** as ranks $r > 500$ (among items with $c(x) > 0$). Compute the global threshold `f_binned_global` (and `f_strict_global`), `max_pass_rate`, and the fraction of items in each regime that satisfy the $95\%$ trial criterion (`pct_head_passing` and `pct_tail_passing`). Panel B (`pct_head_passing` vs. `pct_tail_passing`) is the primary Head/Tail result. Do not compute regime-specific frequency thresholds (`f_strict_head`, `f_binned_head`, `f_strict_tail`, `f_binned_tail`). If a global threshold is undefined, record it as `NaN`; do not substitute any other value.

* **Deterministic RNG Seeds:**

  * Experiment (a) Real Corpus: `seed = 202601 + round(eps * 100000) + d`

  * Experiment (a) Synthetic $s = 1.0$: `seed = 202701 + round(eps * 100000) + d`

  * Experiment (b) Skew Sweep: `seed = 202801 + round(s * 100) + round(eps * 100000) + d`

## 5. Ordered Implementation Stages (Grouped into 3 Agent Sessions)

Work is organized into **3 agent sessions** (Session A: Stages 1–3, Session B: Stages 4–6, Session C: Stage 7 + optional static demo) gated by **3 mandatory human review checkpoints** (Section 10a).

---

### SESSION A: Core Implementation & Verification (Stages 1–3)

#### Stage 1: Data Acquisition, Cleaning, and Zipf Generation

* **Inputs:** Project Gutenberg URLs for eBooks `28054`, `2554`, `600`, `8117`.

* **Outputs:** Cached raw text files in `data/raw/`, processed corpus and seeded multinomial Zipf count vectors saved to `data/processed/streams.npz`, and printed summary table of $N$, $V$, top-10 words, and rank counts.

* **Files Touched:** `requirements.txt`, `data.py`, `tests/test_data.py`.

* **Acceptance Check:**

  ```
  pytest tests/test_data.py -v && python data.py
  ```

#### Stage 2: Count-Min Sketch and Carter-Wegman Hashing + Unit Tests

* **Inputs:** Exact count vector `exact_counts` of shape `(V,)`, width $w$, depth $d$, RNG instance.

* **Outputs:** `CarterWegmanHash` and `CountMinSketch` classes (or vectorized functions) in `sketch.py` supporting single-trial construction via `np.bincount` and batched multi-trial execution. Trials must be processed in chunks of at most $250$ trials per batch, so that no array of shape $(T, V)$ is ever materialized for the full $T = 5000$ (reason: a $(5000 \times V)$ `int64` array is about $1\text{ GB}$ per intermediate).

* **Files Touched:** `sketch.py`, `tests/test_sketch.py`.

* **Acceptance Check:**

  ```
  pytest tests/test_sketch.py -v
  ```

* **PAUSE FOR REVIEW GATE (i):** Stop after `sketch.py` and `tests/test_sketch.py` pass so the user can audit hash, `bincount`, `min` over rows, and verify that the tiny-case test is not vacuous.

#### Stage 3: Trial Harness and Clopper-Pearson CI Utilities

* **Inputs:** `sketch.py` functions, representative item selector (20 Head, 20 Middle, 20 Tail ranks), `scipy.stats.binomtest`.

* **Outputs:** Vectorized/batched trial runner in `experiments.py` that processes trials in chunks of at most $250$ trials per batch (so no $(T, V)$ array is materialized for $T = 5000$) and executes $T = 5000$ independent hash draws for a stream in under $30$ seconds per `(stream, eps, d)` setting on a laptop (if missed, STOP and report per Rule 6; do not restructure the algorithm), plus per-item Clopper-Pearson CI helper and threshold helper.

* **Files Touched:** `experiments.py`, `tests/test_experiments.py`.

* **Acceptance Check:**

  ```
  pytest tests/test_experiments.py -v
  ```

* **PAUSE FOR REVIEW GATE (ii) (End of Session A):** Stop after Stage 3 tests pass so the user can verify that hash parameters are redrawn per trial and Clopper-Pearson CIs use $n = T$ per item.

---

### SESSION B: Experiments & Ablation (Stages 4–6)

*(Use $T = 500$ during initial development checks if needed; all final CSV outputs must use $T = 5000$.)*

#### Stage 4: Experiment (a) — Empirical Failure Probability vs. True Frequency

* **Inputs:** Real Dostoevsky count vector and synthetic Zipf ($s = 1.0$) count vector; $\epsilon \in \{0.01, 0.003, 0.001\}$; $d = 3$ ($\delta = 0.05$); $T = 5000$. Evaluate all 3 classes (`head`, `middle`, `tail`, 20 items each) at primary $\epsilon = 0.003$, and `head` and `tail` only at secondary $\epsilon \in \{0.01, 0.001\}$.

* **Outputs:** `data/processed/exp_a_results.csv` containing columns: `dataset`, `eps`, `w`, `d`, `delta_theory`, `class` (`head`/`middle`/`tail`), `rank`, `item_id`, `word`, `true_count`, `failures_k`, `trials_T`, `fail_prob`, `ci_low`, `ci_high`, `mean_abs_error`, `max_abs_error`. Print compact raw numbers table per Rule 7.

* **Files Touched:** `experiments.py`.

* **Acceptance Check:**

  ```
  python experiments.py --stage exp_a && test -f data/processed/exp_a_results.csv
  ```

* **PAUSE FOR REVIEW GATE (iii):** Print the Stage 4 raw failure count table and STOP. If $k = 0$ everywhere at primary $\epsilon = 0.003$, the user will decide on $\epsilon$ before continuing to Stage 5.

#### Stage 5: Experiment (b) — Relative-Error Threshold vs. Zipf Skew $s$ (Reduced Scope)

* **Inputs:** Synthetic Zipf count vectors for $s \in \{0.6, 0.8, 1.0, 1.2, 1.5\}$ plus Real Dostoevsky corpus; primary $\epsilon = 0.003$ **only** ($w = 907, d = 3$); $T = 5000$.

* **Outputs:**

  * `data/processed/exp_b_thresholds.csv` with columns: `dataset`, `s`, `eps`, `w`, `d`, `f_binned_global`, `f_strict_global`, `pct_head_passing`, `pct_tail_passing`, `rank_of_f_binned`, `max_pass_rate` (the highest mean pass rate $\bar{q}(f)$ over observed frequencies, or over bins). If no frequency satisfies the $95\%$ criterion (binned or strict), record the threshold as `NaN` (do not substitute any other value). Panel B (`pct_head_passing` vs. `pct_tail_passing`) is the primary Head/Tail result.

  * `data/processed/exp_b_curves.csv` with log-binned frequency vs. empirical pass rate curves for each $s$.

  * Print compact raw numbers table per Rule 7.

* **Files Touched:** `experiments.py`.

* **Acceptance Check:**

  ```
  python experiments.py --stage exp_b && test -f data/processed/exp_b_thresholds.csv
  ```

#### Stage 6: $d = 1$ Ablation Study

* **Inputs:** Real Dostoevsky corpus and Synthetic Zipf ($s = 1.0$); $\epsilon \in \{0.01, 0.003, 0.001\}$; $d \in \{1, 3\}$; $T = 5000$. Evaluate all 3 classes (`head`, `middle`, `tail`) at primary $\epsilon = 0.003$, and `head` and `tail` only at secondary $\epsilon \in \{0.01, 0.001\}$.

* **Outputs:** `data/processed/ablation_d1_results.csv` comparing $d = 1$ vs. $d = 3$ empirical failure probabilities, $95\%$ Clopper-Pearson CIs, and mean overestimation $\mathbb{E}[\hat{c}(x) - c(x)] / N$ across items. Print compact raw numbers table per Rule 7.

* **Files Touched:** `experiments.py`.

* **Acceptance Check:**

  ```
  python experiments.py --stage ablation && test -f data/processed/ablation_d1_results.csv
  ```

---

### SESSION C: Figures & Optional Static Demo (Stage 7)

#### Stage 7: Publication-Quality Figures (PNG Only)

* **Inputs:** CSV files in `data/processed/`.

* **Outputs:**

  * `figures/fig1_failure_prob.png` (PNG only, no PDF): Empirical failure probability across Head, Middle (at $\epsilon = 0.003$), and Tail items with $95\%$ Clopper-Pearson error bars vs. theoretical $\delta = 0.05$ (and $1/e$ for $d = 1$), shown across $\epsilon \in \{0.01, 0.003, 0.001\}$ and comparing $d = 3$ vs. $d = 1$.

  * `figures/fig2_rel_error_threshold.png` (PNG only, no PDF): At $\epsilon = 0.003, d = 3$, Panel A showing the global minimum true frequency threshold $f^*$ (`f_binned_global` as primary curve, plus `f_strict_global` if different) vs. Zipf exponent $s \in \{0.6, 0.8, 1.0, 1.2, 1.5\}$ (with horizontal reference line for the Dostoevsky corpus); Panel B showing the fraction of Head ($r \le 500$) vs. Tail ($r > 500$) items achieving $\le 10\%$ relative error in $\ge 95\%$ of trials (`pct_head_passing` and `pct_tail_passing`) across $s$. Panel B is the primary Head/Tail result. In Figure 2, mark any `NaN` thresholds explicitly (e.g. an `"x"` at the top of the axis with the label `"no item passes"`). Plotting code must not crash or silently drop them.

  * *(Optional if time permits)* `demo.ipynb`: Minimal static notebook with at most 3 cells calling a plain function `run_demo(w=907, s=1.0)` (using $T = 500$ or $1000$). No `ipywidgets`.

* **Files Touched:** `plots.py` (and optionally `demo.ipynb`).

* **Acceptance Check:**

  ```
  python plots.py && test -f figures/fig1_failure_prob.png && test -f figures/fig2_rel_error_threshold.png
  ```

*(Note: Former Stage 8 `ipywidgets` notebook and Stage 9 `report/REPORT.md` skeleton are cut. Leave the CSVs in `data/processed/` and PNGs in `figures/` ready for the student's report writing.)*

## 6. Required Automated Tests (Trimmed to Essentials)

Implement only the following essential tests in `tests/` (overflow test and downloaded-file header test are dropped):

1. **Mock-String Header Stripping & Corpus $N, V$ Range (`tests/test_data.py`):**

   * Verify on synthetic mock strings that the stripper removes all text before and including `*** START OF` and all text after and including `*** END OF`, and that tokenized output contains only characters in `[a-z']+`.

   * Print and assert on the processed Dostoevsky corpus that total tokens $N \in [800000, 1300000]$ and vocabulary size $V \in [20000, 40000]$ (include code comment and plan note: `"These ranges are estimates. If an assertion fails, print the actual N and V and STOP and ask (Rule 6); never edit the range yourself."`).

2. **One-Sided Error Guarantee (`tests/test_sketch.py`):**

   * Verify that for **every** item $x \in \{0, \dots, V-1\}$, $\hat{c}(x) \ge c(x)$ without exception.

3. **$d = 1$ Manual-Loop Equivalence (`tests/test_sketch.py`):**

   * For $d = 1$, verify that the sketch's estimate array matches a manual Python loop that sums `exact_counts[x]` into `bucket[h(x)]` and looks up `bucket[h(x)]` for each $x$.

4. **Hand-Checkable Tiny Case ($V = 5, w = 3, d = 2$, `tests/test_sketch.py`):**

   * Hardcode exact counts $c = [100, 50, 20, 5, 1]$ ($N = 176$) for item IDs $x = [0, 1, 2, 3, 4]$, width $w = 3$, depth $d = 2$, and prime $p = 2147483647$:

     * **Row 0:** $a_0 = 2, b_0 = 1 \implies h_0(x) = [1, 0, 2, 1, 0]$. Bucket sums must equal $[51, 105, 20]$, giving row-0 estimates $[105, 51, 20, 105, 51]$.

     * **Row 1:** $a_1 = 1000000000, b_1 = 500000000 \implies (a_1 x + b_1) \bmod p = [500000000, 1500000000, 352516353, 1352516353, 205032706] \implies h_1(x) = [2, 0, 0, 1, 1]$. Bucket sums must equal $[70, 6, 100]$, giving row-1 estimates $[100, 70, 70, 6, 6]$.

     * **Combined Sketch ($\min$ over rows):** Point query estimates for $x = [0, 1, 2, 3, 4]$ must exactly equal $[100, 51, 20, 6, 6]$.

5. **Per-Item Clopper-Pearson CIs and Undefined Thresholds (`tests/test_experiments.py`):**

   * Verify that the CI function returns one interval $(L_i, U_i)$ per item for $n = T$, never pools $n = 20 \cdot T$, and matches `scipy.stats.binomtest(k, T).proportion_ci(confidence_level=0.95, method="exact")` for $k = 0, T = 5000$ ($L = 0.0, U \approx 0.0007375$) and $k = 250, T = 5000$ ($L \approx 0.04412, U \approx 0.05641$).

   * Verify that on a synthetic $\bar{q}$ curve where nothing reaches $0.95$, the threshold computation returns `NaN` for the threshold (`f_binned_global` / `f_strict_global`).

## 7. Workflow Rules for the Agent

1. **Three-Session Execution:** Work is batched into 3 agent sessions rather than 9 separate conversations:
   * **Session A:** Stages 1–3 (`data.py`, `sketch.py`, `experiments.py` harness & CI utilities, and `tests/`).
   * **Session B:** Stages 4–6 (Experiment (a), reduced Experiment (b), and $d=1$ ablation).
   * **Session C:** Stage 7 (`plots.py` generating PNGs, plus optional 3-cell static `demo.ipynb`).
   At the start of each session, re-read `PLAN.md` and inspect existing repo files.

2. **Test-Gate Before Proceeding:** After writing or modifying code for a stage, run that stage's exact **Acceptance Check** command. Use $T = 500$ during development/debugging and $T = 5000$ for the final saved runs.

3. **Three Mandatory Review Gates (Stop Points):** Pause execution and wait for explicit user approval **only** at the three review gates defined in Section 10a:
   * **Gate (i):** After `sketch.py` and `tests/test_sketch.py` are written and passing (end of Stage 2).
   * **Gate (ii):** After the trial harness and `tests/test_experiments.py` are written and passing (end of Stage 3 / Session A).
   * **Gate (iii):** After Stage 4 raw numbers are printed (if $k = 0$ everywhere at $\epsilon = 0.003$, wait for user decision on $\epsilon$ before running Stage 5).
   At the end of each session (or review gate), print a 3–5 bullet summary and commit to git with message `'stage N: <name>'`.

4. **Strict Dependency Allowlist:** Only use Python standard library modules (`re`, `math`, `urllib`, `pathlib`, `argparse`, `unittest`) and the following packages: `numpy`, `scipy`, `matplotlib`, `pandas`, `pytest` (plus `jupyter` only if generating the optional static 3-cell `demo.ipynb`, and at most one optional package such as `probables` strictly inside `tests/` for cross-checking). No `ipywidgets`.

5. **Zero Scope Creep:** Do not add CLI menus, logging frameworks, YAML config parsers, object-oriented class hierarchies beyond a minimal sketch wrapper, or extra statistical tests not specified in this plan.

6. **Infeasible or Wrong Decisions:** If a locked decision looks infeasible or wrong (e.g. an $N$/$V$ assertion fails, or the $30\text{ s}$ runtime target is missed), STOP and ask me. Do not change the decision, loosen an assertion, or work around it silently.

7. **Raw Numbers Before Plotting:** After Stages 4, 5, and 6, print a compact table of raw numbers BEFORE any plotting: per frequency class, failures $k$, trials $T$, and `fail_prob` (min/median/max); per Zipf $s$, the global thresholds (`f_binned_global`, `f_strict_global`), `pct_head_passing`, `pct_tail_passing`, and `max_pass_rate`.

## 8. Out of Scope

Do **not** implement any of the following:

* Space-Saving algorithm, Count Sketch, or Count-Min with Conservative Update.

* Interactive `ipywidgets` sliders or dashboard UIs (if `demo.ipynb` is created at all, it must be at most 3 static cells calling `run_demo(w=907, s=1.0)`).

* Generating `report/REPORT.md` skeleton or placeholders (the student writes the report directly).

* Secondary $\epsilon \in \{0.01, 0.001\}$ sweeps in Experiment (b), regime-specific Head/Tail frequency thresholds (`f_strict_head`, `f_binned_head`, `f_strict_tail`, `f_binned_tail`), or `middle` class items at secondary $\epsilon \in \{0.01, 0.001\}$ in Experiment (a) and the $d=1$ ablation.

* Deterministic `np.round(N * p)` Zipf vectors, PDF figure exports, overflow unit tests, or downloaded-file header-stripping unit tests.

* Token-by-token streaming loops across $N \approx 10^6$ tokens inside Monte Carlo trial loops.

## 9. Cut-List (Priority Order if Time Runs Short)

The 10-hour scope reduction has already cut Space-Saving, `ipywidgets`, `REPORT.md` generation, regime-specific thresholds, Exp (b) secondary $\epsilon$ sweeps, and `middle` items at secondary $\epsilon$. If time still runs short within the 10-hour budget:

1. **Cut 1st:** Omit `demo.ipynb` entirely (keep `run_demo(w=907, s=1.0)` as a plain function in `plots.py` or `experiments.py`).

2. **Cut 2nd:** Drop `f_strict_global` from Figure 2 Panel A and plot only `f_binned_global`.

**NEVER CUT:**

* The $d = 1$ ablation study.

* The per-item $95\%$ Clopper-Pearson confidence intervals ($n = T$).

* The real four-novel Dostoevsky corpus evaluation.

## 10. What I Must Review Myself (Human Checklist)

Before submitting the project, I (the student) will personally inspect and complete the following:

1. **Line-by-Line Code Audit of `sketch.py` (Gate i):** Read every line of the Carter-Wegman hash function $h_r(x) = ((a_r x + b_r) \bmod p) \bmod w$ and the `np.bincount` row construction to verify `np.int64` casting, correct modulo parenthesization, $\min$ aggregation over rows, and that the tiny-case test ($V=5, w=3, d=2$) is not vacuous.

2. **Line-by-Line Audit of `experiments.py` Harness (Gate ii):** Verify that hash parameters $(a_r, b_r)$ are redrawn independently for every trial $t \in \{1, \dots, T\}$, that Clopper-Pearson CIs use $n = T$ per item (not pooled across the 20 items in a class), and that relative error divides by $c(x) > 0$.

3. **Sanity-Check Stage 4 Raw Numbers (Gate iii):** Inspect raw failure counts $k$ across Head, Middle, and Tail items; if $k = 0$ everywhere at $\epsilon = 0.003$, decide whether to adjust $\epsilon$ before running Stage 5.

4. **Write the Report Directly (`2.5h` budget):** Write the final report directly from the generated CSV tables in `data/processed/` and PNG figures in `figures/`, connecting the empirical results to Markov's inequality, collision expectations $\mathbb{E}[\text{error}] = \frac{N - c(x)}{w}$, and the effect of taking the minimum across $d$ rows.

## 10a. Mandatory Review Gates

The agent must pause and wait for my approval at **only** these three checkpoints:

| Gate | After | I check |
| ---- | ----- | ------- |
| **(i)** | Stage 2 (`sketch.py` + `test_sketch.py`) | `sketch.py` line by line (hash, `bincount`, `min` over rows); the tiny-case test is not vacuous |
| **(ii)** | Stage 3 (`experiments.py` harness + `test_experiments.py`) | hash params redrawn per trial; CIs use `n=T` per item, not pooled |
| **(iii)** | Stage 4 raw numbers print | raw failure counts `k`; if `k=0` everywhere at `eps=0.003`, I decide on `eps` before continuing to Stage 5 |

## 11. Open Questions (Resolved with Explicit Defaults)

1. **How Figure 2 represents Head vs. Tail in Reduced Experiment (b):**

   * *Resolution:* Stage 5 runs strictly at primary $\epsilon = 0.003$ ($w = 907, d = 3$). Figure 2 Panel A plots the global threshold `f_binned_global` (and `f_strict_global` if different) vs. Zipf exponent $s \in \{0.6, 0.8, 1.0, 1.2, 1.5\}$ with the Dostoevsky corpus reference line and explicit `"x"` markers ("no item passes") for any `NaN` values. Figure 2 Panel B is the primary Head/Tail result, plotting `pct_head_passing` ($r \le 500$) vs. `pct_tail_passing` ($r > 500$) across $s$. Regime-specific frequency thresholds are dropped.

2. **Synthetic Zipf Stream Construction:**

   * *Resolution:* Stage 1 generates only the seeded multinomial draw `rng_zipf.multinomial(N, p)` (re-sorted descending) as the synthetic stream for each $s$, naturally motivating the rule that relative error is only computed for $c(x) > 0$.