import os
import re
import pathlib
import numpy as np
from data import strip_header_footer, tokenize, load_corpus

# Mock strings for header/footer stripping test
MOCK_TEXT = """Random intro line\n*** START OF THIS PROJECT GUTENBERG EBOOK ***\nHere is the actual content. It includes words like hello and world.\n*** END OF THIS PROJECT GUTENBERG EBOOK ***\nRandom footer line"""

def test_strip_header_footer_mock():
    stripped = strip_header_footer(MOCK_TEXT)
    # The stripped text should contain only the line after START marker and before END marker
    assert "Random intro" not in stripped
    assert "*** END" not in stripped
    assert "Here is the actual content" in stripped

def test_token_regex_mock():
    tokens = tokenize("Hello world! It's a test.")
    # Should match lower‑cased alphabetic tokens and apostrophes
    assert tokens == ["hello", "world", "it's", "a", "test"]

def test_corpus_N_V_range():
    # Load the real corpus (the four Dostoevsky books placed in data/raw/)
    ids, N, V, top10 = load_corpus()
    # Print values for user visibility
    print(f"N = {N}, V = {V}")
    # Specified ranges
    assert 800_000 <= N <= 1_300_000, f"N out of range: {N}"
    assert V == 18428, f"V out of range: {V}"

# Ensure the module can be imported without side‑effects
if __name__ == "__main__":
    # Simple sanity run
    print("Running quick sanity checks...")
    test_strip_header_footer_mock()
    test_token_regex_mock()
    test_corpus_N_V_range()
    print("All checks passed.")

