import numpy as np
from sketch import CarterWegmanHash, CountMinSketch

# Helper to build exact_counts from IDs and frequencies

def build_exact_counts(ids, counts, V):
    """Return a frequency vector of length V where index = ID and value = count.
    ids and counts are parallel arrays.
    """
    freq = np.zeros(V, dtype=np.int64)
    for i, c in zip(ids, counts):
        freq[i] = c
    return freq


def test_one_sided_error_guarantee():
    # Load the real corpus to get IDs and counts
    from data import load_corpus
    ids, N, V, _ = load_corpus()
    # Build exact frequency vector
    exact_counts = np.bincount(ids, minlength=V).astype(np.int64)
    # Sketch parameters (primary config)
    w = 907  # from PLAN (epsilon=0.003)
    d = 3    # depth
    sketch = CountMinSketch(w=w, d=d, seed=12345)
    sketch.build(ids, exact_counts)
    # Estimate for every possible ID
    all_ids = np.arange(V, dtype=np.int64)
    estimates = sketch.estimate(all_ids)
    # One‑sided error: estimate must be >= true count for every item
    assert np.all(estimates >= exact_counts), "One‑sided error violated"


def test_d1_manual_loop_equivalence():
    # Small synthetic example
    V = 5
    ids = np.arange(V, dtype=np.int64)
    exact_counts = np.array([100, 50, 20, 5, 1], dtype=np.int64)
    w = 3
    d = 1
    # Use deterministic hash parameters for reproducibility
    a_vals = np.array([2], dtype=np.int64)
    b_vals = np.array([1], dtype=np.int64)
    sketch = CountMinSketch(w=w, d=d, a_vals=a_vals, b_vals=b_vals)
    sketch.build(ids, exact_counts)
    # Manual loop version
    # Compute hash for each ID
    h = CarterWegmanHash(w, d, a_vals=a_vals, b_vals=b_vals).hash_one(ids)[0]
    bucket = np.bincount(h, weights=exact_counts, minlength=w).astype(np.int64)
    manual_estimates = bucket[h]
    # Sketch estimates (should be same as manual)
    sketch_estimates = sketch.estimate(ids)
    assert np.array_equal(manual_estimates, sketch_estimates), "d=1 manual loop mismatch"


def test_hand_checkable_tiny_case():
    # Tiny case from the plan
    V = 5
    ids = np.arange(V, dtype=np.int64)
    exact_counts = np.array([100, 50, 20, 5, 1], dtype=np.int64)
    w = 3
    d = 2
    # Deterministic hash parameters for the two rows
    a_vals = np.array([2, 1000000000], dtype=np.int64)
    b_vals = np.array([1, 500000000], dtype=np.int64)
    sketch = CountMinSketch(w=w, d=d, a_vals=a_vals, b_vals=b_vals)
    sketch.build(ids, exact_counts)
    # Expected estimates from the specification
    expected = np.array([100, 51, 20, 6, 6], dtype=np.int64)
    estimates = sketch.estimate(ids)
    assert np.array_equal(estimates, expected), f"Tiny case estimates differ: {estimates}"

# Allow running the file directly for a quick sanity check (won't replace pytest execution)
if __name__ == "__main__":
    test_one_sided_error_guarantee()
    test_d1_manual_loop_equivalence()
    test_hand_checkable_tiny_case()
    print("All Count‑Min Sketch tests passed.")

