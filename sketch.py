import numpy as np

# Carter-Wegman universal hash family
class CarterWegmanHash:
    """Implements a family of Carter‑Wegman hash functions for Count‑Min Sketch.
    Each row r has its own (a_r, b_r) parameters drawn uniformly from the
    appropriate ranges. The hash function is:
        h_r(x) = ((a_r * x + b_r) % p) % w
    where p = 2**31 - 1 (a Mersenne prime).
    """

    PRIME = 2_147_483_647  # 2**31 - 1

    def __init__(self, w: int, d: int, seed: int | None = None, a_vals: np.ndarray | None = None, b_vals: np.ndarray | None = None):
        self.w = int(w)
        self.d = int(d)
        rng = np.random.default_rng(seed)
        if a_vals is None:
            # a ∈ {1,…,p‑1}
            self.a = rng.integers(1, self.PRIME, size=self.d, dtype=np.int64)
        else:
            self.a = np.array(a_vals, dtype=np.int64)
        if b_vals is None:
            # b ∈ {0,…,p‑1}
            self.b = rng.integers(0, self.PRIME, size=self.d, dtype=np.int64)
        else:
            self.b = np.array(b_vals, dtype=np.int64)
        # pre‑compute a_mod = a % p, b_mod = b % p for safety (though a,b already in range)
        self.a_mod = self.a % self.PRIME
        self.b_mod = self.b % self.PRIME

    def hash(self, ids: np.ndarray) -> np.ndarray:
        """Vectorised hash for all rows.
        Args:
            ids: 1‑D array of integer item IDs (dtype int64).
        Returns:
            A 2‑D array of shape (d, len(ids)) where each row r contains h_r(ids).
        """
        ids = ids.astype(np.int64)
        # Compute ((a * x + b) % p) % w for each row.
        # Use broadcasting: (d,1) * (1,N) -> (d,N)
        prod = (self.a_mod[:, None] * ids[None, :]) % self.PRIME
        hashed = (prod + self.b_mod[:, None]) % self.PRIME
        return hashed % self.w

    def hash_one(self, ids: np.ndarray) -> np.ndarray:
        """Vectorized hash for an array of IDs.
        Args:
            ids: 1‑D array of item IDs (dtype int64).
        Returns:
            A 2‑D array of shape (d, len(ids)) where each row contains the hash values
            for the corresponding row's (a, b) parameters.
        """
        ids = ids.astype(np.int64)
        # Compute ((a * x + b) % p) % w for each row and each id using broadcasting.
        prod = (self.a_mod[:, None] * ids[None, :]) % self.PRIME
        hashed = (prod + self.b_mod[:, None]) % self.PRIME
        return hashed % self.w


class CountMinSketch:
    """Count‑Min Sketch using the Carter‑Wegman hash family.
    The sketch is built from an exact frequency vector (counts per ID) –
    no token‑by‑token streaming inside trials (linearity guarantee).
    """

    def __init__(self, w: int, d: int, seed: int | None = None, a_vals: np.ndarray | None = None, b_vals: np.ndarray | None = None):
        self.w = w
        self.d = d
        self.hash_family = CarterWegmanHash(w, d, seed=seed, a_vals=a_vals, b_vals=b_vals)
        # Table shape (d, w) initialised to zeros
        self.table = np.zeros((d, w), dtype=np.int64)

    def build(self, ids: np.ndarray, exact_counts: np.ndarray) -> None:
        """Populate the sketch from the exact frequency vector.
        Args:
            ids: 1‑D array of token‑level IDs (ignored – kept for signature compatibility).
            exact_counts: 1‑D array of counts per vocabulary ID (length V).
        """
        # Use vocabulary IDs ranging from 0 to V‑1
        vocab_ids = np.arange(len(exact_counts), dtype=np.int64)
        exact_counts = exact_counts.astype(np.int64)
        # Compute hash values for each row over the vocabulary IDs
        hashes = self.hash_family.hash(vocab_ids)  # shape (d, V)
        # For each row, accumulate counts into buckets via np.bincount
        for r in range(self.d):
            row_hashes = hashes[r]
            bucket = np.bincount(row_hashes, weights=exact_counts, minlength=self.w)
            self.table[r, :] = bucket.astype(np.int64)

    def estimate(self, query_ids: np.ndarray) -> np.ndarray:
        """Return the Count‑Min estimate for each query ID.
        The estimate is the minimum counter across rows for the hashed bucket.
        """
        query_ids = query_ids.astype(np.int64)
        # Hash each query ID for all rows
        hashes = self.hash_family.hash_one(query_ids)  # shape (d, len(query_ids))
        # Gather counters from the table
        # Use fancy indexing: table[r, hashes[r, i]]
        # We can vectorise by flattening.
        d, n = hashes.shape
        # Gather values: for each row r, look up table[r, hashes[r, :]]
        values = np.empty((d, n), dtype=np.int64)
        for r in range(d):
            values[r] = self.table[r, hashes[r]]
        # Minimum across rows gives the estimate
        return np.min(values, axis=0)

    # Convenience method that builds from raw IDs and a frequency vector derived elsewhere
    def fit(self, ids: np.ndarray, exact_counts: np.ndarray) -> None:
        """Alias for build – kept for readability in tests."""
        self.build(ids, exact_counts)

    def query(self, ids: np.ndarray) -> np.ndarray:
        """Alias for estimate – query the sketch for a set of IDs."""
        return self.estimate(ids)

