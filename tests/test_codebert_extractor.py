"""
CodeBERT Feature Extractor Test Suite (Phase 3)
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN
"""

import sys
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.features.codebert_extractor import CodeBERTExtractor


def test_codebert_embedding_dimension(tmp_path):
    """Verify CodeBERT extractor produces exactly 768-dimensional L2-normalized vector."""
    extractor = CodeBERTExtractor(cache_dir=tmp_path / "embeddings")
    code = "def binary_search(arr, x):\n    low, high = 0, len(arr) - 1\n    return -1\n"
    
    result = extractor.extract_embedding(code)
    emb = result["embedding"]
    
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (768,)
    assert emb.dtype == np.float32
    # Verify L2 normalization: norm should be approximately 1.0
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-3)
    assert result["extraction_time_ms"] > 0.0


def test_codebert_sliding_window_long_code(tmp_path):
    """Verify handling of long source code (> 512 tokens) via chunked pooling."""
    extractor = CodeBERTExtractor(cache_dir=tmp_path / "embeddings")
    # Generate long code exceeding 512 tokens
    long_code = "\n".join([f"var_{i} = calculate_value({i}) * {i}" for i in range(300)])
    
    result = extractor.extract_embedding(long_code)
    emb = result["embedding"]
    assert emb.shape == (768,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)


def test_codebert_caching(tmp_path):
    """Verify caching speeds up repeated embedding calls for identical code."""
    extractor = CodeBERTExtractor(cache_dir=tmp_path / "embeddings")
    code = "x = 42\nprint(x * 2)\n"

    # First call: computes embedding and writes cache
    res1 = extractor.extract_embedding(code, use_cache=True)
    assert res1["cached"] is False

    # Second call: reads from cache
    res2 = extractor.extract_embedding(code, use_cache=True)
    assert res2["cached"] is True
    np.testing.assert_array_almost_equal(res1["embedding"], res2["embedding"], decimal=5)
    # Cache lookup should be faster
    assert res2["extraction_time_ms"] <= res1["extraction_time_ms"] + 10.0


def test_batch_extraction(tmp_path):
    """Verify batch extraction returns correct matrix shape (N, 768)."""
    extractor = CodeBERTExtractor(cache_dir=tmp_path / "embeddings")
    snippets = [
        "a = 10\nb = 20\n",
        "def square(x):\n    return x * x\n",
        "for i in range(10):\n    print(i)\n"
    ]
    batch_matrix = extractor.extract_batch(snippets)
    assert isinstance(batch_matrix, np.ndarray)
    assert batch_matrix.shape == (3, 768)
