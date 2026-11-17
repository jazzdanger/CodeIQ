"""
Phase 4 Model Architecture & Similarity Engine Test Suite
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
from src.models.ann_model import MultiTaskCodeANN
from src.similarity.similarity_engine import CodeSimilarityEngine
from src.models.baselines import BaselineModels


def test_ann_model_forward_and_output_shapes():
    """Verify MT-ANN forward pass outputs correct multi-task tensor dimensions."""
    model = MultiTaskCodeANN(input_dim=793)
    batch_size = 4
    x = torch.randn(batch_size, 793)

    outputs = model(x)

    assert "time_logits" in outputs
    assert "space_logits" in outputs
    assert "quality_score" in outputs
    assert "shared_representation" in outputs

    assert outputs["time_logits"].shape == (batch_size, 7)
    assert outputs["space_logits"].shape == (batch_size, 4)
    assert outputs["quality_score"].shape == (batch_size, 1)

    # Verify quality score is strictly bounded in [0.0, 100.0]
    scores = outputs["quality_score"].detach().numpy()
    assert np.all(scores >= 0.0) and np.all(scores <= 100.0)


def test_ann_model_gradient_backpropagation():
    """Verify all 3 task loss gradients propagate through the shared backbone."""
    model = MultiTaskCodeANN(input_dim=793)
    x = torch.randn(2, 793)
    y_time = torch.tensor([0, 3], dtype=torch.long)
    y_space = torch.tensor([1, 0], dtype=torch.long)
    y_qual = torch.tensor([85.0, 72.0], dtype=torch.float32)

    outputs = model(x)
    loss = (
        torch.nn.functional.cross_entropy(outputs["time_logits"], y_time) +
        torch.nn.functional.cross_entropy(outputs["space_logits"], y_space) +
        torch.nn.functional.smooth_l1_loss(outputs["quality_score"].squeeze(-1), y_qual)
    )

    loss.backward()

    # Check that backbone weights have non-zero gradients
    for param in model.input_proj.parameters():
        assert param.grad is not None
        assert not torch.all(param.grad == 0)


def test_ann_model_predict_helper():
    """Verify predict helper decodes valid class indices and confidence scores."""
    model = MultiTaskCodeANN(input_dim=793)
    x = torch.randn(3, 793)

    preds = model.predict(x)

    assert preds["time_class_idx"].shape == (3,)
    assert preds["space_class_idx"].shape == (3,)
    assert preds["quality_score"].shape == (3,)
    assert np.all(preds["time_confidence"] >= 0.0) and np.all(preds["time_confidence"] <= 1.0)


def test_code_similarity_engine():
    """Verify similarity engine compares two submissions and computes 3-tier scores."""
    engine = CodeSimilarityEngine()
    code_a = "def add(x, y):\n    return x + y\n"
    code_b = "def add(a, b):\n    return a + b\n"
    code_c = "class MatrixMultiplication:\n    def multiply(self, A, B):\n        pass\n"

    # Compare identical/near-identical code
    res_similar = engine.compare_submissions(code_a, code_b)
    assert 0.0 <= res_similar["composite_similarity"] <= 100.0
    assert res_similar["composite_similarity"] > 75.0
    assert "disclaimer" in res_similar

    # Compare completely different code
    res_diff = engine.compare_submissions(code_a, code_c)
    assert res_diff["composite_similarity"] < res_similar["composite_similarity"]
