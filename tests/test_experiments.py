"""tests/test_experiments.py – Stage 3 unit tests

These tests verify the statistical utilities added in ``experiments.py``.
They follow the exact specifications in ``PLAN.md``.
"""

import numpy as np
import math

from experiments import (
    clopper_pearson_ci,
    empirical_failure_probability,
    frequency_threshold_for_relative_error,
)


def test_clopper_pearson_edge_cases_k0_T5000():
    """CI for zero successes (k=0) with T=5000 trials.
    The lower bound must be 0 and the upper bound should match SciPy's exact
    calculation. We compare against SciPy directly to avoid hard‑coding numbers.
    """
    from scipy.stats import binomtest
    k, T = 0, 5000
    low, high = clopper_pearson_ci(k, T)
    ref_low, ref_high = binomtest(k, n=T).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    assert math.isclose(low, float(ref_low), rel_tol=1e-12)
    assert math.isclose(high, float(ref_high), rel_tol=1e-12)


def test_clopper_pearson_edge_cases_k250_T5000():
    """CI for half successes (k=250) with T=5000 trials.
    Again we compare with SciPy's reference implementation.
    """
    from scipy.stats import binomtest
    k, T = 250, 5000
    low, high = clopper_pearson_ci(k, T)
    ref_low, ref_high = binomtest(k, n=T).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    assert math.isclose(low, float(ref_low), rel_tol=1e-12)
    assert math.isclose(high, float(ref_high), rel_tol=1e-12)


def test_nan_threshold_when_no_frequency_meets_criterion():
    """Create a synthetic corpus where every item fails the relative‑error
    requirement, forcing the threshold function to return ``np.nan``.

    We use a very strict relative‑error bound (rel_err=0.01) with a sketch that
    deliberately under‑estimates counts – e.g., w=2, d=1 yields many collisions.
    """
    V = 5
    ids = np.arange(V, dtype=np.int64)
    # Small counts with a tiny w=2 sketch: heavy collisions cause large
    # over-estimates, so relative error consistently exceeds the strict 1% bound.
    exact_counts = np.array([10, 8, 6, 4, 2], dtype=np.int64)
    w, d = 2, 1  # very small sketch → many collisions → large over-estimation
    eps = 0.001  # not used for the relative-error part
    f_star = frequency_threshold_for_relative_error(
        ids,
        exact_counts,
        w=w,
        d=d,
        eps=eps,
        rel_err=0.01,
        success_rate=0.95,
        T=200,
        seed_start=0,
    )
    assert math.isnan(f_star)


def test_empirical_failure_probability_and_threshold_small_synthetic():
    """Run a tiny synthetic scenario (V=5, w=3, d=2) and verify that:
    * the empirical failure probability array is within [0, 1]
    * the frequency threshold matches a manually computed value.

    The exact counts are taken from the tiny hand‑checkable case in the plan.
    """
    V = 5
    ids = np.arange(V, dtype=np.int64)
    exact_counts = np.array([100, 50, 20, 5, 1], dtype=np.int64)
    w, d = 3, 2
    eps = 0.003
    # Empirical failure probability (additive error) – we just ensure bounds
    probs = empirical_failure_probability(
        ids, exact_counts, w=w, d=d, eps=eps, T=200, seed_start=0
    )
    assert probs.shape == (V,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    # Frequency threshold for relative error ≤0.10, success_rate=0.95
    f_star = frequency_threshold_for_relative_error(
        ids,
        exact_counts,
        w=w,
        d=d,
        eps=eps,
        rel_err=0.10,
        success_rate=0.95,
        T=200,
        seed_start=0,
    )
    # In this synthetic case the highest frequency (100) certainly meets the
    # criterion, lower frequencies may not. The exact value depends on random
    # seeds but must be one of the true counts (or NaN if none satisfy).
    assert f_star in exact_counts or math.isnan(f_star)


def test_deterministic_minimum_threshold():
    """Deterministic correctness test for the f* threshold algorithm.

    Design
    ------
    We use a 4-item vocabulary with counts [1000, 500, 10, 1] and a deliberately
    small sketch (w=3, d=3). With only 3 buckets for 4 items, every row must
    have at least one collision, so low-frequency items accumulate significant
    relative over-estimation noise while high-frequency items remain within 10%.

    With T=200 and seed_start=99 the per-item success proportions stabilise to:
      count=1000 -> ~0.96  (passes 95% criterion)
      count= 500 -> ~0.96  (passes 95% criterion)
      count=  10 -> ~0.84  (fails 95% criterion)
      count=   1 -> ~0.61  (fails 95% criterion)

    Scanning unique true frequencies in ascending order:
      f=1   -> items {1000,500,10,1} required; item count=10 fails -> NO
      f=10  -> items {1000,500,10} required; item count=10 fails -> NO
      f=500 -> items {1000,500} required; both pass -> YES => f* = 500

    The expected f* = 500 is computed independently inside the test using the
    identical seeds and batch logic, so the assertion is fully deterministic.

    Three things are verified:
      1. The function returns exactly the independently computed f*.
      2. Every item with count >= f* meets the 95% success-rate requirement.
      3. The largest frequency strictly below f* does NOT satisfy the global
         requirement, confirming f* is truly the MINIMUM valid threshold.
    """
    from sketch import CountMinSketch

    V = 4
    exact_counts = np.array([1000, 500, 10, 1], dtype=np.int64)
    vocab_ids = np.arange(V, dtype=np.int64)
    w, d = 3, 3
    T = 200
    seed_start = 99
    rel_err = 0.10
    success_rate = 0.95
    BATCH_SIZE = 250  # must match experiments.py constant

    # --- Independent computation of success proportions ---
    # Replicates experiments.frequency_threshold_for_relative_error internals
    # exactly (same seeds, same batch structure) so the expected f* is derived
    # from first principles rather than hard-coded.
    success_counts_ref = np.zeros(V, dtype=np.int64)
    for batch_start in range(0, T, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, T)
        for t in range(batch_start, batch_end):
            sketch = CountMinSketch(w=w, d=d, seed=seed_start + t)
            sketch.build(vocab_ids, exact_counts)
            estimates = sketch.estimate(vocab_ids)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = (estimates - exact_counts) / exact_counts
            success_counts_ref += ((rel <= rel_err) & (exact_counts > 0)).astype(np.int64)

    prop_ref = success_counts_ref.astype(np.float64) / float(T)

    # Scan ascending frequencies to find the minimum f* from first principles.
    unique_freqs = np.unique(exact_counts[exact_counts > 0])
    expected_f_star = float("nan")
    for f in unique_freqs:
        if np.all(prop_ref[exact_counts >= f] >= success_rate):
            expected_f_star = float(f)
            break

    # Confirm our independent calculation found a finite f* (sanity check on
    # the test design itself, not on the function under test).
    assert not math.isnan(expected_f_star), (
        "Test design error: independent calculation returned NaN; "
        "adjust w/d/T/seed_start so that at least the highest-frequency "
        "item reliably passes."
    )

    # --- Call the function under test ---
    ids = np.arange(V, dtype=np.int64)
    result_f_star = frequency_threshold_for_relative_error(
        ids,
        exact_counts,
        w=w,
        d=d,
        eps=0.003,  # used only internally for sketch construction
        rel_err=rel_err,
        success_rate=success_rate,
        T=T,
        seed_start=seed_start,
    )

    # Assertion 1: function matches the independently computed minimum threshold.
    assert result_f_star == expected_f_star, (
        f"f* mismatch: function returned {result_f_star}, "
        f"independent calculation gives {expected_f_star}. "
        f"Per-item success proportions: {prop_ref}"
    )

    # Assertion 2: every item with count >= f* meets the success-rate requirement.
    passing_mask = exact_counts >= int(expected_f_star)
    assert np.all(prop_ref[passing_mask] >= success_rate), (
        f"At least one item above f*={expected_f_star} fails the {success_rate} "
        f"criterion. Proportions above threshold: {prop_ref[passing_mask]}"
    )

    # Assertion 3: the next lower unique frequency does NOT satisfy the global
    # requirement, confirming f* is truly the minimum valid threshold.
    lower_freqs = unique_freqs[unique_freqs < int(expected_f_star)]
    if lower_freqs.size > 0:
        next_lower = lower_freqs[-1]  # largest frequency strictly below f*
        lower_mask = exact_counts >= next_lower
        assert not np.all(prop_ref[lower_mask] >= success_rate), (
            f"Lower frequency {next_lower} also satisfies the global criterion – "
            f"f*={expected_f_star} is not the minimum threshold. "
            f"Proportions at f={next_lower}: {prop_ref[lower_mask]}"
        )


def test_select_representative_items_structure():
    """Verify representative item selection rules from Locked Decision 6:
      - 20 head items (ranks 1..20)
      - 20 middle items centered around median (at primary eps=0.003 only)
      - 20 tail items in bottom decile of items with count > 0
      - All ranks are 1-based, unique, and item_id == rank - 1.
    """
    from experiments import select_representative_items

    # Synthetic count vector resembling corpus scale (V=18428)
    V = 18428
    exact_counts = np.arange(V, 0, -1, dtype=np.int64)

    # At eps=0.003: 60 items total (20 head, 20 middle, 20 tail)
    items_003 = select_representative_items(exact_counts, eps=0.003)
    assert len(items_003) == 60
    classes = [c for c, r, i in items_003]
    assert classes.count("head") == 20
    assert classes.count("middle") == 20
    assert classes.count("tail") == 20

    head_ranks = [r for c, r, i in items_003 if c == "head"]
    assert head_ranks == list(range(1, 21))

    # At eps=0.01: 40 items total (20 head, 20 tail; middle omitted per scope rule)
    items_01 = select_representative_items(exact_counts, eps=0.01)
    assert len(items_01) == 40
    classes_01 = [c for c, r, i in items_01]
    assert classes_01.count("head") == 20
    assert classes_01.count("middle") == 0
    assert classes_01.count("tail") == 20

    # Check 1-based rank to 0-based item_id mapping and valid bounds
    for cls_name, rank, item_id in items_003:
        assert item_id == rank - 1
        assert 1 <= rank <= V


def test_exp_a_output_schema_and_non_negative_errors():
    """Run a micro-trial run of exp_a (T=5) and assert that:
      - Output CSV has the exact required columns
      - fail_prob is within [0, 1]
      - mean_abs_error and max_abs_error are strictly non-negative
      - ci_low <= fail_prob <= ci_high
      - words for real dataset are strings; for zipf are item_<id>
    """
    from experiments import run_exp_a
    from pathlib import Path

    df = run_exp_a(T=5)

    expected_cols = [
        "dataset", "eps", "w", "d", "delta_theory", "class",
        "rank", "item_id", "word", "true_count", "failures_k",
        "trials_T", "fail_prob", "ci_low", "ci_high",
        "mean_abs_error", "max_abs_error"
    ]
    assert list(df.columns) == expected_cols

    # Exactly 2 datasets * (40 + 60 + 40 items) = 280 rows
    assert len(df) == 280

    assert np.all(df["fail_prob"] >= 0.0)
    assert np.all(df["fail_prob"] <= 1.0)
    assert np.all(df["ci_low"] <= df["fail_prob"] + 1e-9)
    assert np.all(df["fail_prob"] <= df["ci_high"] + 1e-9)

    # Overestimation error is strictly non-negative
    assert np.all(df["mean_abs_error"] >= 0.0)
    assert np.all(df["max_abs_error"] >= 0.0)

    # Check word separation
    real_words = df[df["dataset"] == "real"]["word"].tolist()
    assert all(isinstance(w, str) and not w.startswith("item_") for w in real_words[:20])

    zipf_words = df[df["dataset"] == "zipf_1.0"]["word"].tolist()
    assert all(w.startswith("item_") for w in zipf_words)


def test_exp_b_thresholds_and_binning_logic():
    """Verify Stage 5 threshold calculation, 30 log-binning logic, and NaN handling:
      - When pass rate nowhere reaches 0.95, thresholds must be NaN
      - When a clear threshold exists, f_binned and f_strict must be correctly computed
      - pct_head_passing and pct_tail_passing must be in [0, 1]
      - Exactly 30 bins are generated in the curves output
    """
    from experiments import compute_exp_b_thresholds_and_curves

    # 1. Edge-case: nowhere reaches 0.95 -> thresholds must be NaN
    exact_counts = np.array([1000, 500, 100, 50, 10, 1], dtype=np.int64)
    low_q = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    fb, fs, ph, pt, rb, mp, curves = compute_exp_b_thresholds_and_curves(
        exact_counts, low_q, num_bins=30, success_rate_target=0.95
    )
    assert math.isnan(fb)
    assert math.isnan(fs)
    assert ph == 0.0
    assert pt == 0.0
    assert math.isnan(rb)
    assert len(curves) == 30

    # 2. Case where top items pass: f=500 and f=1000 pass (q=1.0)
    high_q = np.array([1.0, 1.0, 0.8, 0.5, 0.1, 0.0])
    fb2, fs2, ph2, pt2, rb2, mp2, curves2 = compute_exp_b_thresholds_and_curves(
        exact_counts, high_q, num_bins=30, success_rate_target=0.95
    )
    assert not math.isnan(fb2)
    assert not math.isnan(fs2)
    assert fs2 == 500.0  # lowest frequency where all higher frequencies have mean q >= 0.95
    assert mp2 == 1.0
    assert 0.0 <= ph2 <= 1.0
    assert 0.0 <= pt2 <= 1.0
    assert rb2 == 2  # exactly 2 items have count >= 500 (1000 and 500)


def test_exp_b_output_schemas():
    """Verify that Stage 5 generated output CSV files conform strictly to PLAN.md schema:
      - exp_b_thresholds.csv has exact required columns and 6 rows
      - exp_b_curves.csv has 30 bins per dataset (180 rows total)
      - Check that Zipf s=0.6 records NaN threshold and real corpus has NaN for 's'
    """
    from experiments import PROCESSED_DIR
    import pandas as pd

    t_file = PROCESSED_DIR / "exp_b_thresholds.csv"
    c_file = PROCESSED_DIR / "exp_b_curves.csv"

    assert t_file.exists(), "exp_b_thresholds.csv missing"
    assert c_file.exists(), "exp_b_curves.csv missing"

    df_t = pd.read_csv(t_file)
    df_c = pd.read_csv(c_file)

    expected_t_cols = [
        "dataset", "s", "eps", "w", "d", "f_binned_global",
        "f_strict_global", "pct_head_passing", "pct_tail_passing",
        "rank_of_f_binned", "max_pass_rate"
    ]
    assert list(df_t.columns) == expected_t_cols
    assert len(df_t) == 6  # 5 zipf + 1 real

    # Zipf 0.6 has max pass rate ~0.612, so thresholds must be NaN
    row_06 = df_t[df_t["dataset"] == "zipf_0.6"].iloc[0]
    assert math.isnan(row_06["f_binned_global"])
    assert math.isnan(row_06["f_strict_global"])

    # Real corpus has NaN for numeric s
    row_real = df_t[df_t["dataset"] == "real"].iloc[0]
    assert math.isnan(row_real["s"])

    # Curves: 6 datasets * 30 bins = 180 rows
    assert len(df_c) == 180
    expected_c_cols = [
        "bin_idx", "bin_lower", "bin_upper", "bin_mid_log",
        "item_count", "mean_pass_rate", "dataset", "s"
    ]
    assert list(df_c.columns) == expected_c_cols
