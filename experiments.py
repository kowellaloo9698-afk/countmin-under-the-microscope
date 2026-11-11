"""experiments.py – Stage 3 implementation

Provides utilities for running independent Count-Min Sketch trials and
computing statistical metrics required by the project plan.

The functions are deliberately lightweight and use only NumPy and SciPy
as allowed by the locked dependencies.
"""

import numpy as np
from typing import Tuple

from sketch import CountMinSketch

BATCH_SIZE = 250  # maximum trials per batch as required by PLAN.md


def empirical_failure_probability(
    ids: np.ndarray,
    exact_counts: np.ndarray,
    w: int,
    d: int,
    eps: float,
    T: int = 500,
    seed_start: int = 0,
) -> np.ndarray:
    """Return the empirical failure probability of the CMS additive-error guarantee.

    For each vocabulary item the function runs T independent sketches (different
    seeds) and records how often the estimate exceeds the true count by more than
    eps * N where N is the total token count.  The result is a float array of
    shape (V,) containing failures / T for each item.

    Parameters
    ----------
    ids : np.ndarray
        Token-level IDs (kept for API compatibility – the function hashes
        vocabulary IDs internally).
    exact_counts : np.ndarray
        1-D array of true frequencies for each vocabulary ID (length V).
    w, d : int
        Sketch width and depth.
    eps : float
        Additive-error tolerance (as a fraction of N).
    T : int, optional
        Number of independent sketches to run (default 500 for development).
    seed_start : int, optional
        Base seed; each trial uses seed_start + t.

    Returns
    -------
    np.ndarray
        Empirical failure probability per vocabulary item.
    """
    N = int(exact_counts.sum())
    V = exact_counts.shape[0]
    failures = np.zeros(V, dtype=np.int64)

    # Vocabulary IDs are 0 .. V-1; this is what the sketch expects.
    vocab_ids = np.arange(V, dtype=np.int64)

    # Process trials in batches of at most BATCH_SIZE to avoid large arrays.
    for batch_start in range(0, T, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, T)
        for t in range(batch_start, batch_end):
            seed = seed_start + t
            sketch = CountMinSketch(w=w, d=d, seed=seed)
            # ids argument is kept for signature compatibility.
            sketch.build(vocab_ids, exact_counts)
            estimates = sketch.estimate(vocab_ids)
            # Failure if additive error exceeds eps * N.
            fail = (estimates - exact_counts) > (eps * N)
            failures += fail.astype(np.int64)

    return failures.astype(np.float64) / float(T)


def clopper_pearson_ci(
    k: int,
    T: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Exact Clopper-Pearson confidence interval for a binomial proportion.

    Uses scipy.stats.binomtest with method="exact" as mandated by the plan.

    Parameters
    ----------
    k : int
        Number of successes (failures in the trial context).
    T : int
        Number of trials.
    confidence : float, optional
        Desired confidence level (default 0.95).

    Returns
    -------
    (lower, upper) : Tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    from scipy.stats import binomtest
    result = binomtest(k, n=T)
    low, high = result.proportion_ci(confidence_level=confidence, method="exact")
    return float(low), float(high)


def frequency_threshold_for_relative_error(
    ids: np.ndarray,
    exact_counts: np.ndarray,
    w: int,
    d: int,
    eps: float,
    rel_err: float = 0.10,
    success_rate: float = 0.95,
    T: int = 500,
    seed_start: int = 0,
) -> float:
    """Compute the minimum true-frequency f* such that at least success_rate
    fraction of trials achieve relative error <= rel_err for all items with
    frequency >= f*.

    Follows the threshold definition in PLAN.md section 3 (Locked Decision 7).
    Returns np.nan when no frequency satisfies the requirement.

    Parameters
    ----------
    ids : np.ndarray
        Token-level IDs (kept for API compatibility).
    exact_counts : np.ndarray
        True frequencies per vocabulary ID.
    w, d : int
        Sketch dimensions.
    eps : float
        Additive-error tolerance; used only for building the sketch.
    rel_err : float, optional
        Maximum allowed relative error (default 0.10).
    success_rate : float, optional
        Desired proportion of successful trials (default 0.95).
    T : int, optional
        Number of independent trials (default 500).
    seed_start : int, optional
        Base seed for reproducibility.

    Returns
    -------
    float
        The threshold f* or np.nan if undefined.
    """
    V = exact_counts.shape[0]
    success_counts = np.zeros(V, dtype=np.int64)
    vocab_ids = np.arange(V, dtype=np.int64)

    # Process trials in batches of at most BATCH_SIZE.
    for batch_start in range(0, T, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, T)
        for t in range(batch_start, batch_end):
            seed = seed_start + t
            sketch = CountMinSketch(w=w, d=d, seed=seed)
            sketch.build(vocab_ids, exact_counts)
            estimates = sketch.estimate(vocab_ids)
            # Relative error is only defined for items with true count > 0.
            # CMS can only over-estimate (one-sided guarantee), so the relative
            # error is always >= 0 for valid items.
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = (estimates - exact_counts) / exact_counts
            success = (rel <= rel_err) & (exact_counts > 0)
            success_counts += success.astype(np.int64)

    # Per-item proportion of successful trials.
    prop = success_counts.astype(np.float64) / float(T)

    # Only consider items with a positive true count.
    positive_mask = exact_counts > 0
    if not np.any(positive_mask):
        return float("nan")

    unique_freqs = np.unique(exact_counts[positive_mask])

    # Scan unique true frequencies in ascending order.
    # The first f where ALL items with count >= f meet the success-rate
    # criterion is the minimum threshold f*.
    for f in unique_freqs:
        mask = exact_counts >= f
        if np.all(prop[mask] >= success_rate):
            return float(f)

    return float("nan")


import argparse
import collections
from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"


def get_vocabulary_words() -> list[str]:
    """Retrieve the sorted vocabulary list for the real Dostoevsky corpus.
    Sort ordering: descending frequency, ties broken alphabetically.
    Matches load_corpus() in data.py exactly.
    """
    from data import RAW_DIR, strip_header_footer, tokenize
    all_tokens = []
    for path in sorted(RAW_DIR.glob("*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            tokens = tokenize(strip_header_footer(f.read()))
            all_tokens.extend(tokens)
    counter = collections.Counter(all_tokens)
    sorted_items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in sorted_items]


def select_representative_items(
    exact_counts: np.ndarray,
    eps: float,
) -> list[tuple[str, int, int]]:
    """Select representative items based on 1-based frequency ranks.

    Returns a list of tuples: (class_name, 1_based_rank, 0_based_item_id).

    Locked Decision 6:
      - Head (20 items): ranks 1..20
      - Middle (20 items): np.linspace(V // 2 - 95, V // 2 + 95, 20, dtype=int)
        (only evaluated at primary eps=0.003)
      - Tail (20 items): np.linspace(int(0.90 * V_pos), V_pos, 20, dtype=int)
        where V_pos is the number of items with c(x) > 0.
    """
    V = len(exact_counts)
    V_pos = int((exact_counts > 0).sum())

    head_ranks = np.arange(1, 21, dtype=int)
    tail_ranks = np.linspace(int(0.90 * V_pos), V_pos, 20, dtype=int)

    items = []
    for r in head_ranks:
        items.append(("head", int(r), int(r - 1)))

    if np.isclose(eps, 0.003):
        mid_ranks = np.linspace(V // 2 - 95, V // 2 + 95, 20, dtype=int)
        for r in mid_ranks:
            items.append(("middle", int(r), int(r - 1)))

    for r in tail_ranks:
        items.append(("tail", int(r), int(r - 1)))

    return items


def run_exp_a(T: int = 5000) -> pd.DataFrame:
    """Execute Experiment (a): Empirical Failure Probability vs. True Frequency.

    Evaluates the real Dostoevsky corpus and synthetic Zipf (s=1.0) stream across
    eps in {0.01, 0.003, 0.001} with d=3 (delta_theory=0.05).
    Evaluates Head, Middle, Tail at eps=0.003; Head and Tail only at eps in {0.01, 0.001}.

    Results are saved to data/processed/exp_a_results.csv and returned as a DataFrame.
    """
    streams_path = PROCESSED_DIR / "streams.npz"
    if not streams_path.exists():
        raise FileNotFoundError(f"Missing {streams_path}; run data.py first.")

    streams = np.load(streams_path)
    real_counts = streams["real"].astype(np.int64)
    zipf_counts = streams["zipf_1.0"].astype(np.int64)

    real_vocab = get_vocabulary_words()

    configs = [
        {"dataset": "real", "counts": real_counts, "seed_base_offset": 202601},
        {"dataset": "zipf_1.0", "counts": zipf_counts, "seed_base_offset": 202701},
    ]

    eps_sweep = [
        {"eps": 0.01, "w": 272},
        {"eps": 0.003, "w": 907},
        {"eps": 0.001, "w": 2719},
    ]

    d = 3
    delta_theory = 0.05

    results_rows = []

    for cfg in configs:
        ds_name = cfg["dataset"]
        counts = cfg["counts"]
        N = int(counts.sum())
        V = len(counts)
        vocab_ids = np.arange(V, dtype=np.int64)

        for ep_info in eps_sweep:
            eps = ep_info["eps"]
            w = ep_info["w"]

            # Deterministic isolated seed per configuration (PLAN.md Section 4.2)
            seed_base = cfg["seed_base_offset"] + round(eps * 100000) + d

            # Select representative items
            rep_items = select_representative_items(counts, eps)
            q_ids = np.array([item[2] for item in rep_items], dtype=np.int64)
            q_counts = counts[q_ids]

            failures_k = np.zeros(len(q_ids), dtype=np.int64)
            sum_abs_err = np.zeros(len(q_ids), dtype=np.float64)
            max_abs_err = np.zeros(len(q_ids), dtype=np.int64)

            # Batched trial loop (chunk size <= BATCH_SIZE=250)
            for b_start in range(0, T, BATCH_SIZE):
                b_end = min(b_start + BATCH_SIZE, T)
                for t in range(b_start, b_end):
                    seed = seed_base + t
                    sk = CountMinSketch(w=w, d=d, seed=seed)
                    sk.build(vocab_ids, counts)
                    estimates = sk.estimate(q_ids)

                    # Non-negative overestimation error
                    err = estimates - q_counts
                    fail = err > (eps * N)

                    failures_k += fail.astype(np.int64)
                    sum_abs_err += err.astype(np.float64)
                    max_abs_err = np.maximum(max_abs_err, err)

            fail_prob = failures_k.astype(np.float64) / float(T)
            mean_abs_err = sum_abs_err / float(T)

            # Record per-item results
            for idx, (cls_name, rank, item_id) in enumerate(rep_items):
                true_c = int(counts[item_id])
                k_val = int(failures_k[idx])
                low_ci, high_ci = clopper_pearson_ci(k_val, T, confidence=0.95)

                if ds_name == "real":
                    word = real_vocab[item_id]
                else:
                    word = f"item_{item_id}"

                results_rows.append({
                    "dataset": ds_name,
                    "eps": eps,
                    "w": w,
                    "d": d,
                    "delta_theory": delta_theory,
                    "class": cls_name,
                    "rank": rank,
                    "item_id": item_id,
                    "word": word,
                    "true_count": true_c,
                    "failures_k": k_val,
                    "trials_T": T,
                    "fail_prob": float(fail_prob[idx]),
                    "ci_low": float(low_ci),
                    "ci_high": float(high_ci),
                    "mean_abs_error": float(mean_abs_err[idx]),
                    "max_abs_error": int(max_abs_err[idx]),
                })

    df = pd.DataFrame(results_rows)
    out_csv = PROCESSED_DIR / "exp_a_results.csv"
    df.to_csv(out_csv, index=False)

    # Print compact raw numbers table (Rule 7)
    print("\n" + "=" * 80)
    print("STAGE 4 RAW NUMBERS SUMMARY: EXPERIMENT (a)")
    print(f"Trials per configuration: T = {T}")
    print("=" * 80)
    summary_cols = ["dataset", "eps", "class"]
    grouped = df.groupby(summary_cols)
    print(f"{'Dataset':<10} {'eps':<8} {'Class':<8} {'Items':<6} {'k (min/med/max)':<18} {'fail_prob (min/med/max)':<26} {'Mean Abs Err (med)':<20}")
    print("-" * 100)
    for (ds, ep, cls), g in grouped:
        k_str = f"{g['failures_k'].min()} / {int(g['failures_k'].median())} / {g['failures_k'].max()}"
        fp_str = f"{g['fail_prob'].min():.4f} / {g['fail_prob'].median():.4f} / {g['fail_prob'].max():.4f}"
        mae_med = f"{g['mean_abs_error'].median():.1f}"
        print(f"{ds:<10} {ep:<8} {cls:<8} {len(g):<6} {k_str:<18} {fp_str:<26} {mae_med:<20}")
    print("=" * 80)
    print(f"Saved results to: {out_csv}\n")

    return df


def compute_relative_error_pass_rates(
    exact_counts: np.ndarray,
    w: int,
    d: int,
    eps: float,
    seed_base: int,
    rel_err: float = 0.10,
    T: int = 500,
) -> np.ndarray:
    """Compute per-item empirical pass rate q(x) across T independent trials.
    Pass criterion: relative overestimation error (estimate - true) / true <= rel_err.
    Only defined for items with true count c(x) > 0.
    Items with c(x) == 0 are assigned pass rate 0.0.
    """
    V = len(exact_counts)
    vocab_ids = np.arange(V, dtype=np.int64)
    pos_mask = exact_counts > 0

    success_counts = np.zeros(V, dtype=np.int64)

    for b_start in range(0, T, BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, T)
        for t in range(b_start, b_end):
            seed = seed_base + t
            sk = CountMinSketch(w=w, d=d, seed=seed)
            sk.build(vocab_ids, exact_counts)
            estimates = sk.estimate(vocab_ids)

            rel = np.zeros(V, dtype=np.float64)
            rel[pos_mask] = (estimates[pos_mask] - exact_counts[pos_mask]) / exact_counts[pos_mask]
            success = (rel <= rel_err) & pos_mask
            success_counts += success.astype(np.int64)

    q = np.zeros(V, dtype=np.float64)
    q[pos_mask] = success_counts[pos_mask].astype(np.float64) / float(T)
    return q


def compute_exp_b_thresholds_and_curves(
    exact_counts: np.ndarray,
    q: np.ndarray,
    num_bins: int = 30,
    success_rate_target: float = 0.95,
) -> tuple[float, float, float, float, int | float, float, list[dict]]:
    """Compute binned and strict global thresholds, head/tail passing fractions,
    max pass rate, and the log-binned curve points for a stream.

    Returns:
        f_binned_global, f_strict_global, pct_head_passing, pct_tail_passing,
        rank_of_f_binned, max_pass_rate, curve_rows
    """
    pos_mask = exact_counts > 0
    if not np.any(pos_mask):
        return np.nan, np.nan, 0.0, 0.0, np.nan, 0.0, []

    c_max = float(exact_counts[pos_mask].max())
    bins = np.logspace(0, np.log10(c_max), num_bins + 1)

    # Digitize positive items into 0-based bin indices [0 .. num_bins-1]
    # np.digitize returns 1-based index where bins[i-1] <= x < bins[i]
    bin_indices = np.digitize(exact_counts[pos_mask], bins) - 1
    # Guard against edge-case where x == bins[-1] (index becomes num_bins)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    bin_means = np.full(num_bins, np.nan)
    bin_counts = np.zeros(num_bins, dtype=int)
    bin_lower = bins[:-1]
    bin_upper = bins[1:]

    pos_q = q[pos_mask]

    curve_rows = []
    for b in range(num_bins):
        in_bin = (bin_indices == b)
        cnt = int(np.sum(in_bin))
        bin_counts[b] = cnt
        if cnt > 0:
            mean_q = float(np.mean(pos_q[in_bin]))
            bin_means[b] = mean_q
        else:
            mean_q = np.nan

        curve_rows.append({
            "bin_idx": b,
            "bin_lower": float(bin_lower[b]),
            "bin_upper": float(bin_upper[b]),
            "bin_mid_log": float(10 ** (0.5 * (np.log10(bin_lower[b]) + np.log10(bin_upper[b])))),
            "item_count": cnt,
            "mean_pass_rate": mean_q,
        })

    # 1. f_binned_global: smallest bin lower edge where all higher-or-equal non-empty bins have mean pass rate >= 0.95
    f_binned_global = np.nan
    for b in range(num_bins):
        if bin_counts[b] == 0:
            continue
        higher_non_empty = [bin_means[k] for k in range(b, num_bins) if bin_counts[k] > 0]
        if all(m >= success_rate_target for m in higher_non_empty):
            f_binned_global = float(bin_lower[b])
            break

    # 2. f_strict_global: smallest observed frequency f such that every observed frequency f' >= f has mean pass rate >= 0.95
    unique_freqs = np.unique(exact_counts[pos_mask])  # sorted ascending
    mean_q_per_freq = {f: float(np.mean(q[exact_counts == f])) for f in unique_freqs}

    f_strict_global = np.nan
    for f in unique_freqs:
        higher_freqs = [fp for fp in unique_freqs if fp >= f]
        if all(mean_q_per_freq[fp] >= success_rate_target for fp in higher_freqs):
            f_strict_global = float(f)
            break

    # 3. Head vs Tail passing percentages (Head: ranks 1..500; Tail: ranks > 500 among c(x) > 0)
    # The arrays are pre-sorted descending by frequency
    V = len(exact_counts)
    ranks = np.arange(1, V + 1)
    head_mask = (ranks <= 500) & pos_mask
    tail_mask = (ranks > 500) & pos_mask

    pct_head_passing = float(np.mean(q[head_mask] >= success_rate_target)) if np.any(head_mask) else 0.0
    pct_tail_passing = float(np.mean(q[tail_mask] >= success_rate_target)) if np.any(tail_mask) else 0.0

    # 4. rank_of_f_binned: 1-based rank corresponding to f_binned_global
    if not np.isnan(f_binned_global):
        rank_of_f_binned = int(np.sum(exact_counts >= f_binned_global))
    else:
        rank_of_f_binned = np.nan

    # 5. max_pass_rate: highest mean pass rate over non-empty bins (or observed frequencies)
    valid_bin_means = [m for m in bin_means if not np.isnan(m)]
    max_pass_rate = float(np.max(valid_bin_means)) if valid_bin_means else 0.0

    return f_binned_global, f_strict_global, pct_head_passing, pct_tail_passing, rank_of_f_binned, max_pass_rate, curve_rows


def run_exp_b(T: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Experiment (b): Relative-Error Threshold vs. Zipf Skew.

    Runs on synthetic Zipf streams for s in {0.6, 0.8, 1.0, 1.2, 1.5} plus the
    real Dostoevsky corpus.
    Parameters: primary configuration eps=0.003, w=907, d=3 only.
    Generates data/processed/exp_b_thresholds.csv and data/processed/exp_b_curves.csv.
    """
    streams_path = PROCESSED_DIR / "streams.npz"
    if not streams_path.exists():
        raise FileNotFoundError(f"Missing {streams_path}; run data.py first.")

    streams = np.load(streams_path)

    eps = 0.003
    w = 907
    d = 3

    # Define datasets to sweep
    # PLAN.md 4.2: seed = 202801 + round(s * 100) + round(eps * 100000) + d
    sweep = [
        {"dataset": "zipf_0.6", "s": 0.6, "counts": streams["zipf_0.6"].astype(np.int64), "seed": 202801 + round(0.6 * 100) + round(eps * 100000) + d},
        {"dataset": "zipf_0.8", "s": 0.8, "counts": streams["zipf_0.8"].astype(np.int64), "seed": 202801 + round(0.8 * 100) + round(eps * 100000) + d},
        {"dataset": "zipf_1.0", "s": 1.0, "counts": streams["zipf_1.0"].astype(np.int64), "seed": 202801 + round(1.0 * 100) + round(eps * 100000) + d},
        {"dataset": "zipf_1.2", "s": 1.2, "counts": streams["zipf_1.2"].astype(np.int64), "seed": 202801 + round(1.2 * 100) + round(eps * 100000) + d},
        {"dataset": "zipf_1.5", "s": 1.5, "counts": streams["zipf_1.5"].astype(np.int64), "seed": 202801 + round(1.5 * 100) + round(eps * 100000) + d},
        {"dataset": "real", "s": np.nan, "counts": streams["real"].astype(np.int64), "seed": 202801 + round(eps * 100000) + d},
    ]

    threshold_rows = []
    all_curve_rows = []

    for item in sweep:
        ds_name = item["dataset"]
        s_val = item["s"]
        counts = item["counts"]
        seed_base = item["seed"]

        # Compute empirical pass rate q(x) across T trials
        q = compute_relative_error_pass_rates(
            exact_counts=counts,
            w=w,
            d=d,
            eps=eps,
            seed_base=seed_base,
            rel_err=0.10,
            T=T,
        )

        f_binned, f_strict, pct_head, pct_tail, rank_binned, max_p, curves = compute_exp_b_thresholds_and_curves(
            exact_counts=counts,
            q=q,
            num_bins=30,
            success_rate_target=0.95,
        )

        threshold_rows.append({
            "dataset": ds_name,
            "s": s_val,
            "eps": eps,
            "w": w,
            "d": d,
            "f_binned_global": f_binned,
            "f_strict_global": f_strict,
            "pct_head_passing": pct_head,
            "pct_tail_passing": pct_tail,
            "rank_of_f_binned": rank_binned,
            "max_pass_rate": max_p,
        })

        for cr in curves:
            cr["dataset"] = ds_name
            cr["s"] = s_val
            all_curve_rows.append(cr)

    df_thresholds = pd.DataFrame(threshold_rows)
    df_curves = pd.DataFrame(all_curve_rows)

    out_thresholds = PROCESSED_DIR / "exp_b_thresholds.csv"
    out_curves = PROCESSED_DIR / "exp_b_curves.csv"

    df_thresholds.to_csv(out_thresholds, index=False)
    df_curves.to_csv(out_curves, index=False)

    # Print compact raw numbers table (Rule 7)
    print("\n" + "=" * 95)
    print("STAGE 5 RAW NUMBERS SUMMARY: EXPERIMENT (b)")
    print(f"Trials per configuration: T = {T} (eps = {eps}, w = {w}, d = {d})")
    print("=" * 95)
    print(f"{'Dataset':<10} {'s':<6} {'f_binned':<14} {'f_strict':<14} {'pct_head':<12} {'pct_tail':<12} {'rank_binned':<14} {'max_pass':<10}")
    print("-" * 95)
    for _, r in df_thresholds.iterrows():
        s_str = f"{r['s']:.1f}" if not np.isnan(r['s']) else "real"
        fb_str = f"{r['f_binned_global']:.1f}" if not np.isnan(r['f_binned_global']) else "NaN"
        fs_str = f"{r['f_strict_global']:.1f}" if not np.isnan(r['f_strict_global']) else "NaN"
        rb_str = f"{int(r['rank_of_f_binned'])}" if not np.isnan(r['rank_of_f_binned']) else "NaN"
        print(f"{r['dataset']:<10} {s_str:<6} {fb_str:<14} {fs_str:<14} {r['pct_head_passing']:<12.4f} {r['pct_tail_passing']:<12.4f} {rb_str:<14} {r['max_pass_rate']:<10.4f}")
    print("=" * 95)
    print(f"Saved thresholds to: {out_thresholds}")
    print(f"Saved curves to:     {out_curves}\n")

    return df_thresholds, df_curves


def main():
    parser = argparse.ArgumentParser(description="CS328 Count-Min Sketch Experiments")
    parser.add_argument("--stage", choices=["exp_a", "exp_b", "ablation"], required=True, help="Stage to execute")
    parser.add_argument("--trials", type=int, default=5000, help="Number of trials per configuration (default 5000, dev 500)")
    args = parser.parse_args()

    if args.stage == "exp_a":
        run_exp_a(T=args.trials)
    elif args.stage == "exp_b":
        run_exp_b(T=args.trials)
    elif args.stage == "ablation":
        raise NotImplementedError("Stage 6 (ablation) is scheduled for later in Session B.")


if __name__ == "__main__":
    main()
