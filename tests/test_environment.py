"""
Environment and Foundation Verification Test Suite (Phase 1)
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN
"""

import sys
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.utils.logger import logger, setup_logger
from src.utils.device import set_seed, get_device, get_hardware_profile, get_gpu_memory_usage
from src.utils.sanitization import CodeSanitizer


def test_python_version():
    """Verify runtime Python meets minimum requirement (>= 3.10)."""
    assert sys.version_info >= (3, 10), f"Python 3.10+ required. Found: {sys.version}"


def test_directory_structure():
    """Ensure all required project directories exist."""
    required_dirs = [
        settings.DATASETS_DIR / "raw",
        settings.DATASETS_DIR / "processed",
        settings.DATASETS_DIR / "embeddings",
        settings.MODELS_DIR / "checkpoints",
        settings.MODELS_DIR / "baselines",
        settings.REPORTS_DIR / "figures",
        settings.REPORTS_DIR / "metrics",
        settings.BASE_DIR / "backend" / "api",
        settings.BASE_DIR / "backend" / "database",
        settings.BASE_DIR / "backend" / "schemas",
        settings.BASE_DIR / "backend" / "services",
        settings.BASE_DIR / "src" / "utils",
        settings.BASE_DIR / "src" / "data",
        settings.BASE_DIR / "src" / "features",
        settings.BASE_DIR / "src" / "models",
        settings.BASE_DIR / "src" / "similarity",
        settings.BASE_DIR / "tests",
        settings.BASE_DIR / "frontend",
    ]
    for directory in required_dirs:
        assert directory.exists(), f"Missing required directory: {directory}"
        assert directory.is_dir(), f"Path is not a directory: {directory}"


def test_package_imports():
    """Verify core third-party dependencies can be imported successfully."""
    import torch
    import transformers
    import radon
    import sklearn
    import fastapi
    import sqlalchemy
    import pydantic
    import networkx

    assert torch.__version__ is not None
    assert transformers.__version__ is not None
    assert radon.__version__ is not None
    assert sklearn.__version__ is not None
    assert fastapi.__version__ is not None
    assert sqlalchemy.__version__ is not None
    assert pydantic.__version__ is not None
    assert networkx.__version__ is not None


def test_hardware_profiling():
    """Verify hardware profile detection and CUDA/GPU status."""
    profile = get_hardware_profile()
    assert "cuda_available" in profile
    assert "pytorch_version" in profile
    assert "gpu_name" in profile
    assert "total_vram_gb" in profile

    device = get_device(settings.DEVICE)
    assert device.type in ("cuda", "cpu")

    # If CUDA is available, check that memory usage functions return valid numbers
    if profile["cuda_available"]:
        mem = get_gpu_memory_usage()
        assert "allocated_mb" in mem
        assert "reserved_mb" in mem
        assert mem["allocated_mb"] >= 0.0


def test_config_feature_definitions():
    """Validate 25 AST feature names and complexity classes in config."""
    assert len(settings.AST_FEATURE_NAMES) == 25, (
        f"Expected exactly 25 AST features, found {len(settings.AST_FEATURE_NAMES)}"
    )
    assert settings.NUM_AST_FEATURES == 25
    assert settings.CODEBERT_EMBEDDING_DIM == 768
    assert settings.TOTAL_FEATURE_DIM == 793

    # Check Time Complexity classes
    assert len(settings.TIME_COMPLEXITY_CLASSES) == 7
    assert "O(1)" in settings.TIME_COMPLEXITY_CLASSES
    assert "O(N log N)" in settings.TIME_COMPLEXITY_CLASSES
    assert "O(2^N)" in settings.TIME_COMPLEXITY_CLASSES

    # Check Space Complexity classes
    assert len(settings.SPACE_COMPLEXITY_CLASSES) == 4
    assert "O(1)" in settings.SPACE_COMPLEXITY_CLASSES
    assert "O(N)" in settings.SPACE_COMPLEXITY_CLASSES


def test_deterministic_seeding():
    """Verify set_seed produces deterministic outputs in random, numpy, and torch."""
    import random
    import numpy as np
    import torch

    set_seed(42)
    val_py_1 = random.random()
    val_np_1 = np.random.rand(5)
    val_pt_1 = torch.rand(5)

    set_seed(42)
    val_py_2 = random.random()
    val_np_2 = np.random.rand(5)
    val_pt_2 = torch.rand(5)

    assert val_py_1 == val_py_2
    np.testing.assert_array_equal(val_np_1, val_np_2)
    assert torch.equal(val_pt_1, val_pt_2)


def test_code_sanitizer():
    """Verify input validation, size limits, and non-executing AST syntax checks."""
    # 1. Valid code
    valid_code = "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n"
    is_valid, msg, meta = CodeSanitizer.validate_submission(valid_code, "python")
    assert is_valid is True
    assert msg == ""
    assert meta["syntax_valid"] is True
    assert meta["line_count"] == 2

    # 2. Syntax error code
    invalid_code = "def broken(x\n    return x +"
    is_valid, msg, meta = CodeSanitizer.validate_submission(invalid_code, "python")
    assert is_valid is False
    assert "SyntaxError" in msg
    assert meta["syntax_valid"] is False

    # 3. Empty submission
    is_valid, msg, _ = CodeSanitizer.validate_submission("", "python")
    assert is_valid is False
    assert "empty" in msg.lower()

    # 4. Unsupported language
    is_valid, msg, _ = CodeSanitizer.validate_submission(valid_code, "rust")
    assert is_valid is False
    assert "unsupported" in msg.lower()

    # 5. Oversized code
    oversized = "a = 1\n" * 15000  # Exceeds 50 KB
    is_valid, msg, _ = CodeSanitizer.validate_submission(oversized, "python")
    assert is_valid is False
    assert "maximum permitted size" in msg
