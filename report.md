# 1. Problem Description

Tracking item frequencies in high-volume data streams is a fundamental requirement in network traffic monitoring, search engine query processing, and natural language analytics. In standard streaming settings, an exact frequency table requires maintaining a dictionary with one entry per unique item, scaling with the vocabulary size as $O(V)$ in memory. When streams comprise millions of tokens and tens of thousands of distinct items, or when counting must operate under strict hardware memory constraints (such as router SRAM or edge caches), maintaining exact state becomes prohibitive.

To address memory constraints, streaming algorithms employ probabilistic sketching. The Count-Min Sketch (Cormode & Muthukrishnan, 2005) maintains a two-dimensional counter table of width $w$ and depth $d$, parameterized by an additive error tolerance $\varepsilon$ and a failure probability ceiling $\delta$. Given stream length $N$, the algorithm guarantees that any point query $\hat{c}(x)$ satisfies:
$$\Pr[\hat{c}(x) - c(x) > \varepsilon N] \le \delta$$
Furthermore, because hash collisions only add non-negative counts to buckets, the Count-Min Sketch provides a one-sided error guarantee: $\hat{c}(x) \ge c(x)$ for all $x$, precluding underestimates.

While this theoretical guarantee provides strong worst-case bounds, two significant questions arise in practical applications:
1. **Empirical Slackness of the Failure Ceiling:** The theoretical bound $\delta \le (1/e)^d$ is derived via Markov's inequality, assuming worst-case hash collisions across arbitrary stream configurations. Real-world streams, such as natural language text, exhibit heavily skewed Zipfian frequency distributions where a small fraction of items accounts for the majority of the token mass. Consequently, empirical failure rates may remain well below $\delta$. Furthermore, the empirical performance gap between single-row ($d=1$) and multi-row ($d=3$) configurations across frequency strata warrants systematic measurement.
2. **Relative-Error Usability for Low-Frequency Items:** The Count-Min guarantee is strictly additive ($\varepsilon N$). For high-frequency items where $c(x) \gg \varepsilon N$, an additive error of $\varepsilon N$ corresponds to negligible relative error. Conversely, for low-frequency and tail items where $c(x) \ll \varepsilon N$, an additive error of scale $\varepsilon N$ can dwarf the true count by several orders of magnitude. In downstream applications requiring accurate percentage bounds, the primary operational metric is the relative error:
$$\text{RelErr}(x) = \frac{\hat{c}(x) - c(x)}{c(x)} \le \theta$$
It is therefore essential to determine the minimum frequency threshold $f^*$ required to ensure that relative error does not exceed $\theta = 0.10$ with probability at least $0.95$, and to evaluate how $f^*$ shifts across stream skewness.

This report investigates two research questions:
- **Research Question 1 (Q1):** How loose is the theoretical failure bound $\Pr[\hat{c}(x) - c(x) > \varepsilon N] \le \delta$ on natural language text, how do empirical failure rates differ across frequency tiers (head, middle, tail), and how does depth ablation ($d=1$ vs. $d=3$) affect estimation error and failure rates?
- **Research Question 2 (Q2):** What is the minimum true frequency $f^*$ required to achieve $\le 10\%$ relative error in $\ge 95\%$ of randomized trials, and how does this threshold change as the Zipf skew exponent $s$ varies?

---

# 2. Solution Approach

### Survey of Existing Methods
Several paradigms exist for frequency estimation in streaming environments:
- **Exact Hash Tables:** Maintain an exact counter for every encountered token. While achieving zero error, exact tables require $O(V)$ space, creating unbounded memory growth in open-vocabulary streams.
- **Counter-Based Summary Algorithms (Misra-Gries, Space-Saving, Lossy Counting):** Track the top-$k$ most frequent items deterministically within $O(k)$ memory by incrementing or decrementing candidate counters. While effective for heavy-hitter discovery, counter-based summaries require pointer overhead, do not provide point queries across arbitrary tail items, and cannot be easily merged across distributed workers without complex reconciliation.
- **Count Sketch (Charikar, Chen, Farach-Colton, 2002):** Employs 2-universal hash functions paired with rademacher random hash functions mapping items to $\{-1, +1\}$. By taking the median of bucket estimates across rows, Count Sketch yields an unbiased estimator whose variance depends on the stream's second frequency moment $F_2$. However, its two-sided error can produce negative estimates, and median computations require wider tables to achieve comparable failure ceilings.
- **Count-Min Sketch (Cormode & Muthukrishnan, 2005):** Allocates a $d \times w$ matrix of counters, where $w = \lceil e / \varepsilon \rceil$ and $d = \lceil \ln(1 / \delta) \rceil$. Each row $j \in \{0, \dots, d-1\}$ uses an independent 2-universal hash function $h_j(x)$ mapping items to $\{0, \dots, w-1\}$. On each arrival of item $x$, counter $C[j, h_j(x)]$ is incremented. The point estimate is computed via the row-wise minimum:
$$\hat{c}(x) = \min_{0 \le j < d} C[j, h_j(x)]$$
Count-Min Sketch was chosen for this study because it provides strictly non-negative, one-sided point estimates with fixed $O(w \times d)$ space and $O(d)$ query and update time. Crucially, its linear additivity enables exact vectorized construction over frequency vectors via weighted histogram aggregation, eliminating the need to replay token-by-token streams during Monte Carlo evaluations.

### Proposed Evaluation Methodology
Our experimental evaluation evaluates Count-Min Sketch under parameter settings $\varepsilon = 0.003$ and $\delta = 0.05$, yielding dimensions $w = 907$ counters and $d = 3$ rows (2,721 total counters). Universal hashing is implemented using the Carter-Wegman hash family:
$$h_{a,b}(x) = ((a \cdot x + b) \bmod p) \bmod w$$
where $p = 2^{31} - 1$ is the Mersenne prime $M_{31}$, with coefficients $a \in [1, p-1]$ and $b \in [0, p-1]$ drawn uniformly at random per row.

The evaluation is conducted on a natural language text stream constructed from four novels by Fyodor Dostoevsky sourced from Project Gutenberg, yielding $N = 870,124$ tokens and a vocabulary of $V = 18,428$ distinct words. The experimental workflow proceeds in two primary phases:
1. **Experiment 1 (Failure Probability & Depth Ablation):** We evaluate empirical failure rates across $T = 5000$ independent Monte Carlo sketch repetitions. A stratified sample of 60 representative words is selected: 20 Head items (ranks 1–20), 20 Middle items (ranks centered around $V/2$), and 20 Tail items (selected from the bottom 10% of positive items). For each item, we measure empirical failure frequency $k = \sum_{t=1}^T \mathbb{I}(\hat{c}_t(x) - c(x) > \varepsilon N)$ and compute exact 95% Clopper-Pearson confidence intervals. Depth ablation compares the $d=3$ sketch against a single-row $d=1$ sketch ($w=907$), and theoretical heavy-hitter collision probabilities are analyzed. Robustness checks verify stability across alternative error tolerances ($\varepsilon \in \{0.01, 0.001\}$) and synthetic Zipfian distributions.
2. **Experiment 2 (Relative Error Thresholds & Skew Sweep):** For each vocabulary item $x \in \{0, \dots, V-1\}$, we evaluate the empirical pass rate $q(x) = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(\text{RelErr}_t(x) \le 0.10)$ over $T = 5000$ trials. Items are partitioned into 30 logarithmically spaced frequency bins to determine the binned threshold $f^*_{\text{binned}}$ (the lower bound of the lowest bin where all subsequent bins maintain mean $q \ge 0.95$). To assess the impact of distributional skew, we repeat this analysis across synthetic Zipf streams of identical length $N$ and vocabulary $V$ with skew exponents $s \in \{0.6, 0.8, 1.0, 1.2, 1.5\}$.

---

# 3. Results

### Methodology Recap
The empirical evaluation was conducted over $T = 5000$ independent Monte Carlo trials evaluated in memory chunks of 250 trials using Carter-Wegman universal hashing modulo the Mersenne prime $p = 2^{31} - 1$. Per-item 95% confidence intervals were computed using the exact Clopper-Pearson method, verified against boundary unit checks: for zero failures ($k = 0, T = 5000$), the method yields an upper bound of $0.0007375$ (exact $0.0007375038011147727$), and for nominal 5% failures ($k = 250, T = 5000$), it yields $[0.04412, 0.05641]$ (exact $[0.044123243979991046, 0.05640891324217884]$). Algorithm correctness was established on a five-item toy corpus ($V_{\text{toy}} = 5, w_{\text{toy}} = 3, d_{\text{toy}} = 2$) with true counts $[100, 50, 20, 5, 1]$: Row 0 ($a=2, b=1$) produced bucket sums $[51, 105, 20]$ with point estimates $[105, 51, 20, 105, 51]$; Row 1 ($a=10^9, b=5\times 10^8$) produced bucket sums $[70, 6, 100]$ with point estimates $[100, 70, 70, 6, 6]$; and the pointwise row minimum resolved final estimates to $[100, 51, 20, 6, 6]$, verifying the one-sided error property ($\hat{c}(x) \ge c(x)$) with zero underestimates.

---

### Experiment 1: Empirical Failure Probability and Depth Ablation

To evaluate the slackness of the theoretical bound $\Pr[\hat{c}(x) - c(x) > \varepsilon N] \le \delta$ ($\varepsilon = 0.003, \delta = 0.05, \varepsilon N = 2610.4$), empirical failure rates were measured across 60 representative words stratified into Head, Middle, and Tail tiers.

#### Class Summary at $d=3$
The empirical failure distributions across the three tiers are summarized below:

| Frequency Class | Ranks Sampled | Min Fail Prob | Median Fail Prob | Max Fail Prob | Zero-Failure Items (%) | Empirical Slack vs. $\delta=0.05$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Head** | $1 - 20$ | $0.0000$ | $0.0000$ | $0.0002$ | $80.0\%$ | $250.0\times$ |
| **Middle** | $9119 - 9309$ | $0.0000$ | $0.0002$ | $0.0008$ | $10.0\%$ | $62.5\times$ |
| **Tail** | $16585 - 18428$ | $0.0000$ | $0.0002$ | $0.0008$ | $30.0\%$ | $62.5\times$ |

Across all 60 evaluated items, empirical failure probabilities remained well below the theoretical ceiling $\delta = 0.05$. For Head items, 16 out of 20 items ($16 / 20 = 80.0\%$) recorded zero failures over 5,000 trials, with a maximum observed failure probability of $0.0002$ (1 failure), corresponding to a slack factor of $\delta / \max(P_{\text{fail}}) = 0.05 / 0.0002 = 250.0\times$. For Middle and Tail items, the maximum observed failure probability was $0.0008$ (4 failures out of 5,000 trials, observed for 'ferment' at rank 9159 and 'pupils' at rank 16682), corresponding to an empirical slack factor of $\delta / \max(P_{\text{fail}}) = 0.05 / 0.0008 = 62.5\times$.

#### Depth Ablation Study: $d=1$ vs. $d=3$
To quantify the impact of taking the minimum across multiple independent hash rows, an ablation study was conducted comparing single-row sketches ($d=1, w=907$) against multi-row sketches ($d=3, w=907$). The results from `data/processed/ablation_d1_results.csv` are presented below:

| Frequency Class | Median $P_{\text{fail}}$ ($d=1$) | Median $P_{\text{fail}}$ ($d=3$) | Mean Overestimate $/ N$ ($d=1$) | Mean Overestimate $/ N$ ($d=3$) | Error Reduction Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Head** | $0.0424$ | $0.0$ | $0.000915172205340848$ | $0.0002283757257586275$ | $4.00730945594499$ |
| **Middle** | $0.062$ | $0.0002$ | $0.0011143439325889185$ | $0.0002656555847212581$ | $4.194694170492051$ |
| **Tail** | $0.0597$ | $0.0002$ | $0.0010796330178227472$ | $0.0002646303285508732$ | $4.079778095484606$ |

At $d=1$, median failure probabilities ranged between $0.0424$ and $0.062$, which remain below the single-row theoretical Markov ceiling of $1/e \approx 0.367879$. Transitioning from $d=1$ to $d=3$ reduced median failure rates by over two orders of magnitude (to $0.0$ for Head and $0.0002$ for Middle and Tail). Across all three classes, taking the pointwise minimum across $d=3$ rows reduced mean normalized overestimation error by an average factor of $4.09\times$ (Head: $4.01\times$, Middle: $4.19\times$, Tail: $4.08\times$).

#### Theoretical Heavy-Hitter Collision Analysis
The observed empirical slack is explained by the distribution of token mass in the stream. An additive error exceeding $\varepsilon N = 2610.4$ requires an item to collide with one or more heavy hitters whose individual count exceeds $\varepsilon N$. In the corpus ($N = 870,124, V = 18,428$), exactly $m = 48$ words have true counts exceeding $2610.4$. 

Under independent universal hashing into $w = 907$ buckets:
- **Single-row ($d=1$):** The theoretical collision probability with any of the $m$ heavy hitters is:
$$\left(\frac{m}{w}\right)^1 = \frac{48}{907} \approx 0.05292$$
This theoretical expectation aligns with the measured mean $d=1$ failure probability across items of $0.05464$.
- **Multi-row ($d=3$):** Because row hashes are independent, an additive violation requires colliding with a heavy hitter in all $d=3$ rows simultaneously:
$$\left(\frac{m}{w}\right)^3 = \left(\frac{48}{907}\right)^3 \approx 0.00014822$$
This joint probability corresponds closely to the measured mean $d=3$ failure probability of $0.00017667$.

#### Robustness Checks
To establish that this slack is not an artifact of specific parameter choices or text, sensitivity evaluations were performed across alternative error tolerances and synthetic streams:
- **Real corpus, $\varepsilon = 0.01$ ($w = 272$):** Mean failure probability = $0.0002$, maximum failure probability = $0.0006$.
- **Real corpus, $\varepsilon = 0.001$ ($w = 2719$):** Mean failure probability = $0.0001$, maximum failure probability = $0.0006$.
- **Synthetic Zipf $s = 1.0$, $\varepsilon = 0.003$ ($w = 907$):** Mean failure probability = $0.0001$, maximum failure probability = $0.0004$.

Across all robustness configurations, the maximum observed failure probability was $0.0006$, remaining substantially below $\delta = 0.05$.

#### Visual Inspection: Figure 1
Figure 1 (`figures/fig1_failure_prob.png`) visualizes empirical failure rates alongside theoretical bounds across the 60 representative items:
- The x-axis indexes items from 0 to 59, divided by vertical dotted lines into three labeled regions: `HEAD (r <= 20)`, `MIDDLE (r ~ V//2)`, and `TAIL (bottom 10%)`.
- Orange squares plot the $d=1$ empirical failure rates, clustering near their empirical mean ($0.055$), safely below the theoretical bound $1/e \approx 0.368$ (indicated by a horizontal line).
- Blue circles with vertical error bars show $d=3$ empirical failure rates with 95% Clopper-Pearson confidence intervals. All $d=3$ points cluster at or below $0.0008$, remaining well below the horizontal dashed line representing $\delta = 0.05$.

---

### Experiment 2: Relative-Error Usability and Frequency Thresholds

To address Q2, we evaluate the frequency required for an item to achieve a relative error $\le 10\%$ in at least $95\%$ of trials ($q(x) \ge 0.95$).

#### Frequency Threshold Sweep across Zipf Skew
The complete set of empirical thresholds from `data/processed/exp_b_thresholds.csv` is presented below:

| `dataset` | `s` | `f_binned` | `f_strict` | `max_pass_rate` | `pct_head_passing` | `pct_tail_passing` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `zipf_0.6` | `0.6` | *(empty)* | *(empty)* | `0.6582` | `0.0` | `0.0` |
| `zipf_0.8` | `0.8` | `7141.897190993186` | `7692.0` | `1.0` | `1.0` | `0.0` |
| `zipf_1.0` | `1.0` | `4073.576138842974` | `4428.0` | `1.0` | `3.8` | `0.0` |
| `zipf_1.2` | `1.2` | `2112.9023166594875` | `2070.0` | `1.0` | `8.200000000000001` | `0.0` |
| `zipf_1.5` | `1.5` | `378.74401023909155` | `418.0` | `1.0` | `17.599999999999998` | `0.0` |
| `real` | *(empty)* | `3067.654458915819` | `3271.0` | `1.0` | `7.3999999999999995` | `0.0` |

#### Non-Convergence at Low Skew ($s=0.6$)
For synthetic stream `zipf_0.6` ($s = 0.6$), the token mass is dispersed across many items rather than concentrated into heavy hitters. As a result, bucket collision noise is distributed broadly across the table. Under these conditions, no frequency bin or item satisfies the 95% pass rate criterion; the maximum pass rate observed across all items was $0.6582$ (65.82%). Consequently, `f_binned` and `f_strict` are undefined (`NaN`), and both Head and Tail passing percentages are $0.0\%$.

#### Threshold Dynamics with Skewness
As skewness increases from $s = 0.8$ to $s = 1.5$:
- The binned threshold $f^*_{\text{binned}}$ decreases monotonically from $7141.897190993186$ ($s = 0.8$) to $4073.576138842974$ ($s = 1.0$), $2112.9023166594875$ ($s = 1.2$), and $378.74401023909155$ ($s = 1.5$). Higher skewness concentrates stream mass into fewer items, leaving the remaining buckets less noisy and allowing lower-frequency items to meet the 10% relative error target.
- For the natural language text corpus (`real`), the binned threshold is $f^*_{\text{binned}} = 3067.654458915819$ (strict threshold $f^*_{\text{strict}} = 3271.0$), situating the natural corpus between Zipf skews $s = 1.0$ and $s = 1.2$.
- In the real corpus, $7.3999999999999995\%$ (7.4%) of Head items ($r \le 500$) meet the 95% reliability criterion, while $0.0\%$ of Tail items ($r > 500$) pass. Across all evaluated datasets, the tail passing rate remained strictly $0.0\%$.

#### Visual Inspection: Figure 2
Figure 2 (`figures/fig2_rel_error_threshold.png`) illustrates these threshold trends:
- **Panel A ("Relative-Error Threshold $f^*$ vs. Skew"):** Plots $f^*$ on a logarithmic scale spanning 200 to 25,000 counts against skew exponent $s$. A solid blue line connects the convergent thresholds from $s=0.8$ to $s=1.5$. A horizontal green dotted line indicates the real corpus threshold ($f^* \approx 3068$). A red 'x' marker at $s=0.6$ (plotted at count 12,000) explicitly marks the non-convergent condition with the annotation `"x: no item passes (max q < 95%)"`.
- **Panel B ("Head vs. Tail Passing Percentage"):** Plots item reliability percentages across skews $s \in [0.6, 1.5]$. The blue line shows Head items ($r \le 500$) increasing from $0.0\%$ at $s=0.6$ to $1.0\%$ ($s=0.8$), $3.8\%$ ($s=1.0$), $8.2\%$ ($s=1.2$), and $17.6\%$ ($s=1.5$), with a green dotted horizontal reference line at $7.4\%$ for the real corpus. The red dashed line representing Tail items ($r > 500$) remains flat at $0.0\%$ across all skews.

---

# 4. Conclusions

### Answers to Research Questions

**Research Question 1 (Empirical Failure Slackness and Depth Ablation):**
The empirical evaluation confirms that the theoretical failure bound $\Pr[\hat{c}(x) - c(x) > \varepsilon N] \le \delta$ ($\delta = 0.05, \varepsilon N = 2610.4$) exhibits substantial slack on natural language text. For Head items, $80.0\%$ of evaluated words incurred zero failures across 5,000 trials, with a maximum failure probability of $0.0002$, corresponding to an empirical slack factor of $250.0\times$ relative to $\delta$. For Middle and Tail items, the maximum observed failure probability was $0.0008$, yielding a slack factor of $62.5\times$. The ablation study demonstrates that increasing sketch depth from $d=1$ to $d=3$ drops median failure rates from between $0.0424$ and $0.0620$ (safely below the single-row theoretical ceiling $1/e \approx 0.368$) down to $0.0$ for Head and $0.0002$ for Middle and Tail items, while reducing mean normalized overestimation error by an average factor of $4.09\times$ ($4.01\times$ Head, $4.19\times$ Middle, $4.08\times$ Tail). This empirical slack is directly explained by the concentration of stream mass into heavy hitters: only $m = 48$ words out of $V = 18,428$ exceed $\varepsilon N = 2610.4$. The theoretical collision probability $(m/w)^1 = 48 / 907 \approx 0.05292$ closely mirrors the measured mean $d=1$ failure probability of $0.05464$, while the joint three-row collision probability $(m/w)^3 \approx 0.00014822$ accurately mirrors the measured mean $d=3$ failure rate of $0.00017667$.

**Research Question 2 (Relative-Error Usability and Distributional Skew):**
The minimum true item frequency $f^*$ required to guarantee $\le 10\%$ relative error in $\ge 95\%$ of randomized trials is strongly dependent on the skewness of the underlying frequency distribution. For the natural language corpus, the binned reliability threshold is $f^*_{\text{binned}} = 3067.654458915819$ ($f^*_{\text{strict}} = 3271.0$). At this threshold, only $7.3999999999999995\%$ (7.4%) of Head items ($r \le 500$) meet the reliability criterion, while $0.0\%$ of Tail items ($r > 500$) pass. Across synthetic Zipf distributions, when skewness is low ($s = 0.6$), dispersion of count mass prevents any item from achieving $95\%$ reliability (maximum observed pass rate is $0.6582$), rendering $f^*$ non-convergent (`NaN`). As skewness increases from $s = 0.8$ to $s = 1.5$, the binned threshold decreases monotonically from $7141.897190993186$ down to $378.74401023909155$, and the proportion of passing Head items expands from $1.0\%$ to $17.599999999999998\%$ (17.6%). Across all evaluated natural and synthetic datasets, the tail passing rate remained strictly $0.0\%$, demonstrating that while Count-Min Sketch provides robust additive guarantees globally, it cannot provide reliable relative error bounds for low-frequency items without scaling table width to the order of $O(1 / (\theta \cdot f_{\text{tail}}))$.

---

### Limitations
- **Static Corpus Evaluation:** The experimental workload relies on concatenated, static literary texts without real-time stream dynamics such as item deletions, out-of-order arrivals, or bursty traffic spikes.
- **Fixed Vocabulary and Stream Length:** Experiments were conducted at a single scale ($N = 870,124, V = 18,428$). The evaluation does not explore empirical performance when vocabulary scales to tens of millions of items, such as large-scale network IP flows or search query logs.
- **Single Hash Family:** Hashing was implemented exclusively using Carter-Wegman linear congruential universal hashing modulo the Mersenne prime $M_{31}$. Non-cryptographic hash functions common in production environments (such as MurmurHash3 or xxHash) were not benchmarked.
- **Absence of Ultra-Low Frequency Testing at Scale:** While tail items were evaluated, the minimum non-zero frequency in the evaluated natural language dataset was 1. Ultra-sparse streams where the ratio $V / N \to 1$ were not tested.

---

### Future Work
- **Comparative Analysis with Count Sketch and Conservative Update:** Benchmarking Count-Min Sketch against Count Sketch (median estimator with $\pm 1$ hashing) and Conservative Update Count-Min Sketch (CU-CMS) to quantify empirical variance reduction and overestimation suppression under identical memory footprints.
- **Time-Decayed and Sliding-Window Sketches:** Extending the evaluation to sliding-window and exponential-decay models to determine how time-decayed streaming affects heavy-hitter concentration and relative-error thresholds.
- **Adversarial and Non-Uniform Stream Robustness:** Investigating sketch degradation under adversarial streaming patterns specifically constructed to induce hash collisions across shared buckets.

---

# 5. Code and Data

### Interactive Execution Notebook
- **Google Colab Notebook:** [Link to Google Colab Notebook](https://colab.research.google.com/) *(Placeholder for user link)*
- **Local Executed Notebook:** [countmin_executed_T5000.ipynb](file:///c:/Users/kowel/Desktop/ids-project/countmin_executed_T5000.ipynb)

### Primary Data Sources (Project Gutenberg)
The natural language evaluation corpus comprises four unabridged novels by Fyodor Dostoevsky retrieved from Project Gutenberg:
1. *The Brothers Karamazov* (Gutenberg eBook ID: `28054`) — [https://www.gutenberg.org/cache/epub/28054/pg28054.txt](https://www.gutenberg.org/cache/epub/28054/pg28054.txt)
2. *Crime and Punishment* (Gutenberg eBook ID: `2554`) — [https://www.gutenberg.org/cache/epub/2554/pg2554.txt](https://www.gutenberg.org/cache/epub/2554/pg2554.txt)
3. *Notes from Underground* (Gutenberg eBook ID: `600`) — [https://www.gutenberg.org/cache/epub/600/pg600.txt](https://www.gutenberg.org/cache/epub/600/pg600.txt)
4. *The Possessed (The Devils)* (Gutenberg eBook ID: `8117`) — [https://www.gutenberg.org/cache/epub/8117/pg8117.txt](https://www.gutenberg.org/cache/epub/8117/pg8117.txt)

### Generated Artifacts
All quantitative results, threshold calculations, and visualizations cited in this report are persisted in the repository as the full set of generated artifacts:
- **Experiment 1 Item Results (60 items):** [data/processed/exp_a_results.csv](file:///c:/Users/kowel/Desktop/ids-project/data/processed/exp_a_results.csv)
- **Experiment 2 Thresholds (6 dataset configurations):** [data/processed/exp_b_thresholds.csv](file:///c:/Users/kowel/Desktop/ids-project/data/processed/exp_b_thresholds.csv)
- **Depth Ablation Results ($d=1$ vs. $d=3$):** [data/processed/ablation_d1_results.csv](file:///c:/Users/kowel/Desktop/ids-project/data/processed/ablation_d1_results.csv)
- **Figure 1 (Empirical Failure Probabilities):** [figures/fig1_failure_prob.png](file:///c:/Users/kowel/Desktop/ids-project/figures/fig1_failure_prob.png)
- **Figure 2 (Relative Error Thresholds vs. Skew):** [figures/fig2_rel_error_threshold.png](file:///c:/Users/kowel/Desktop/ids-project/figures/fig2_rel_error_threshold.png)
