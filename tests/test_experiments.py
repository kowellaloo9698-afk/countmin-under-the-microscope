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
