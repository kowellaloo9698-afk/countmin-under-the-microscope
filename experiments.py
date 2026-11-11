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
