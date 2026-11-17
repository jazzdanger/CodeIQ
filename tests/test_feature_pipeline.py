"""
Feature Pipeline Integration Test Suite (Phase 3)
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN
"""

import sys
import pytest
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.features.feature_pipeline import FeaturePipeline


def test_feature_pipeline_fusion_dimensions(tmp_path):
    """Verify feature pipeline correctly concatenates 25 AST + 768 CodeBERT into 793 dims."""
    scaler_path = tmp_path / "test_scaler.joblib"
    pipeline = FeaturePipeline(scaler_path=scaler_path)

    code = """
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
"""
    result = pipeline.transform_single(code)

    assert "fused_vector" in result
    fused = result["fused_vector"]
    assert isinstance(fused, np.ndarray)
    assert fused.shape == (793,), f"Expected shape (793,), got {fused.shape}"
    assert fused.dtype == np.float32

    # Verify component shapes
    assert result["ast_vector"].shape == (25,)
    assert result["codebert_vector"].shape == (768,)
    assert result["total_time_ms"] > 0.0


def test_feature_pipeline_scaler_persistence(tmp_path):
    """Verify fitting and saving scaler creates reusable artifact."""
    scaler_path = tmp_path / "fitted_scaler.joblib"
    pipeline = FeaturePipeline(scaler_path=scaler_path)

    # Create dummy AST vectors (10 samples, 25 features)
    dummy_ast = np.random.rand(10, 25).astype(np.float32) * 50.0
    pipeline.fit_scaler(dummy_ast)

    assert scaler_path.exists()

    # Re-instantiate pipeline and ensure scaler is loaded
    pipeline_loaded = FeaturePipeline(scaler_path=scaler_path)
    assert pipeline_loaded.scaler is not None
