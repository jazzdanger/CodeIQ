"""
AST Feature Extractor Test Suite (Phase 3)
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
from src.features.ast_extractor import ASTFeatureExtractor


def test_ast_extractor_feature_count():
    """Verify exactly 25 features are produced in the correct order."""
    extractor = ASTFeatureExtractor()
    code = """
def sample_function(x, y):
    total = 0
    for i in range(x):
        for j in range(y):
            if i == j:
                total += i * j
    return total
"""
    result = extractor.extract_features(code)
    assert result["success"] is True
    assert result["error"] is None
    assert len(result["features"]) == 25
    assert isinstance(result["vector"], np.ndarray)
    assert result["vector"].shape == (25,)
    assert result["extraction_time_ms"] > 0.0

    # Check key structural values
    feats = result["features"]
    assert feats["num_functions"] == 1.0
    assert feats["num_loops"] == 2.0
    assert feats["max_loop_nesting_depth"] == 2.0
    assert feats["num_nested_loops"] == 1.0
    assert feats["num_conditional_statements"] == 1.0
    assert feats["num_return_statements"] == 1.0
    assert feats["num_parameters"] == 2.0


def test_recursive_function_detection():
    """Verify recursive function calls are accurately identified."""
    extractor = ASTFeatureExtractor()
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
    result = extractor.extract_features(code)
    assert result["success"] is True
    assert result["features"]["num_recursive_functions"] == 1.0


def test_syntax_error_graceful_handling():
    """Verify syntax errors return a clean zeroed 25-dim vector without crashing."""
    extractor = ASTFeatureExtractor()
    broken_code = "def invalid_syntax(x\n   return x +"
    result = extractor.extract_features(broken_code)

    assert result["success"] is False
    assert result["error"] is not None
    assert "SyntaxError" in result["error"]
    assert result["vector"].shape == (25,)
    assert np.all(result["vector"] == 0.0)


def test_empty_code_handling():
    """Verify empty input returns a clean zeroed 25-dim vector without crashing."""
    extractor = ASTFeatureExtractor()
    result = extractor.extract_features("   \n\n  ")

    assert result["success"] is False
    assert "Empty" in result["error"]
    assert result["vector"].shape == (25,)
    assert np.all(result["vector"] == 0.0)


def test_latency_measurement():
    """Verify latency is recorded in milliseconds and is realistic."""
    extractor = ASTFeatureExtractor()
    code = "x = sum([i**2 for i in range(100)])"
    result = extractor.extract_features(code)
    assert result["extraction_time_ms"] >= 0.0
    assert isinstance(result["extraction_time_ms"], float)
