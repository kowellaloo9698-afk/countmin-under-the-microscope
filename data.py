import re
import os
from pathlib import Path
import collections
import numpy as np

RAW_DIR = Path(__file__).parent / 'data' / 'raw'
PROCESSED_DIR = Path(__file__).parent / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Gutenberg header/footer markers (case‑insensitive)
START_MARK = re.compile(r'^\*\*\* START OF', re.IGNORECASE)
END_MARK = re.compile(r'^\*\*\* END OF', re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z']+")

def strip_header_footer(text: str) -> str:
    """Remove everything before the first line starting with *** START OF and after the last line starting with *** END OF.
    The marker lines themselves are excluded.
    """
    lines = text.splitlines()
    start_idx = 0
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if START_MARK.match(line):
            start_idx = i + 1
            break
    for i in range(len(lines) - 1, -1, -1):
        if END_MARK.match(lines[i]):
            end_idx = i
            break
    return "\n".join(lines[start_idx:end_idx])

def tokenize(text: str) -> list[str]:
    """Return a list of lowercase tokens matching [a‑z']+.
    """
    return TOKEN_RE.findall(text.lower())

def load_corpus() -> tuple[np.ndarray, int, int, list[tuple[str, int]]]:
    """Load the four Gutenberg texts, strip headers/footers, tokenize, and build the ID mapping.
    Returns:
        ids: ndarray of shape (N,) with integer IDs for each token.
        N: total token count.
        V: vocabulary size.
        top10: list of (word, count) for the ten most frequent words.
    """
    all_tokens = []
    for path in sorted(RAW_DIR.glob('*.txt')):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        stripped = strip_header_footer(raw)
        tokens = tokenize(stripped)
        all_tokens.extend(tokens)
    N = len(all_tokens)
    counter = collections.Counter(all_tokens)
    # Determine ordering: descending count, then alphabetical for ties
    sorted_items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = [word for word, _ in sorted_items]
    V = len(vocab)
    # Map word -> ID
    word_to_id = {word: i for i, word in enumerate(vocab)}
    ids = np.array([word_to_id[t] for t in all_tokens], dtype=np.int64)
    top10 = sorted_items[:10]
    return ids, N, V, top10

def generate_zipf_streams(N: int, V: int) -> dict[float, np.ndarray]:
    """For each Zipf exponent s in {0.6,0.8,1.0,1.2,1.5} generate a seeded multinomial count vector.
    The vector is sorted in descending order so that rank order is preserved.
    Returns a dict mapping s to the count vector (np.ndarray of shape (V,)).
    """
    s_values = [0.6, 0.8, 1.0, 1.2, 1.5]
    streams = {}
    ranks = np.arange(1, V + 1, dtype=np.float64)
    for s in s_values:
        # probability for each rank
        probs = ranks ** (-s)
        probs /= probs.sum()
        seed = 1000 + int(s * 100)
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(N, probs)
        # sort descending to keep rank order (largest count first)
        counts_sorted = np.sort(counts)[::-1]
        streams[s] = counts_sorted
    return streams

def save_streams(streams: dict[float, np.ndarray]):
    """Save all Zipf streams and the real corpus counts to a .npz file.
    The real corpus counts are stored under key 'real'.
    """
    # Also save the real corpus counts (exact frequencies per ID)
    ids, N, V, _ = load_corpus()
    # Compute exact counts per ID
    real_counts = np.bincount(ids, minlength=V).astype(np.int64)
    # Prepare dict for np.savez
    npz_dict = {'real': real_counts}
    for s, arr in streams.items():
        npz_dict[f'zipf_{s}'] = arr.astype(np.int64)
    out_path = PROCESSED_DIR / 'streams.npz'
    np.savez(out_path, **npz_dict)
    return out_path

def main():
    ids, N, V, top10 = load_corpus()
    print(f"Total tokens (N): {N}")
    print(f"Vocabulary size (V): {V}")
    print("Top‑10 words:")
    for word, cnt in top10:
        print(f"  {word}: {cnt}")
    streams = generate_zipf_streams(N, V)
    out_path = save_streams(streams)
    print(f"Saved streams to {out_path}")

if __name__ == '__main__':
    main()

