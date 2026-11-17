"""
Dataset Pipeline Verification Test Suite (Phase 2)
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN
"""

import sys
import pytest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.data.taxonomy import PROBLEM_REGISTRY, get_problem_spec
from src.data.preprocess import CodePreprocessor
from src.data.label_generator import LabelGenerator
from src.data.download_dataset import CodeNetDatasetDownloader
from src.data.dataset_builder import DatasetBuilder


def test_taxonomy_completeness():
    """Verify taxonomy covers all required time and space complexity classes."""
    time_classes_in_taxonomy = set()
    space_classes_in_taxonomy = set()

    for pid, spec in PROBLEM_REGISTRY.items():
        assert spec.problem_id == pid
        assert len(spec.sample_solutions) >= 1
        time_classes_in_taxonomy.add(spec.time_complexity)
        space_classes_in_taxonomy.add(spec.space_complexity)

    # Check all 7 time complexity classes are covered
    for tc in settings.TIME_COMPLEXITY_CLASSES:
        assert tc in time_classes_in_taxonomy, f"Missing time complexity class in taxonomy: {tc}"

    # Check all 4 space complexity classes are covered
    for sc in settings.SPACE_COMPLEXITY_CLASSES:
        assert sc in space_classes_in_taxonomy, f"Missing space complexity class in taxonomy: {sc}"


def test_code_preprocessor_validation():
    """Test syntactic validation and anti-DoS size limits."""
    preprocessor = CodePreprocessor()

    # Valid Python sample
    valid_code = "def add(a, b):\n    return a + b\n"
    is_valid, reason, meta = preprocessor.validate_code(valid_code)
    assert is_valid is True
    assert meta["syntax_valid"] is True

    # Syntax Error
    broken_code = "def broken(:\n    pass"
    is_valid, reason, meta = preprocessor.validate_code(broken_code)
    assert is_valid is False
    assert "SyntaxError" in reason

    # Too short
    is_valid, reason, _ = preprocessor.validate_code("x = 1")
    assert is_valid is False
    assert "too short" in reason.lower()

    # Deduplication
    preprocessor.reset_deduplication()
    p1, _, _ = preprocessor.filter_and_deduplicate("s1", valid_code)
    p2, reason2, _ = preprocessor.filter_and_deduplicate("s2", valid_code)
    assert p1 is True
    assert p2 is False
    assert "duplicate" in reason2.lower()


def test_maintainability_index_calculation():
    """Test empirical Software Engineering Maintainability Index calculation via Radon."""
    simple_clean_code = """
def calculate_area(length, width):
    \"\"\"Calculates rectangle area.\"\"\"
    if length <= 0 or width <= 0:
        return 0
    return length * width
"""
    mi_clean = LabelGenerator.compute_maintainability_index(simple_clean_code)
    assert 0.0 <= mi_clean <= 100.0
    assert mi_clean > 50.0  # Simple clean code should have healthy maintainability

    # Nested complex code should have lower maintainability index
    complex_code = """
def complex_routine(data):
    res = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] > 10:
                for k in range(5):
                    if k % 2 == 0 and data[i][j] % 3 == 0:
                        res += data[i][j] * k
                    elif k % 3 == 0:
                        res -= k
    return res
"""
    mi_complex = LabelGenerator.compute_maintainability_index(complex_code)
    assert 0.0 <= mi_complex <= 100.0
    assert mi_complex < mi_clean  # Complex code must have strictly lower MI than simple code


def test_zero_data_leakage_dataset_builder(tmp_path):
    """
    Verify DatasetBuilder creates exact CSV schema and enforces 0% problem ID leakage
    between train, val, and test splits.
    """
    downloader = CodeNetDatasetDownloader(raw_dir=tmp_path / "raw")
    # Generate smaller corpus for rapid unit testing
    corpus = downloader.generate_algorithmic_corpus(target_samples=100, seed=42)
    assert len(corpus) >= 100

    builder = DatasetBuilder(output_dir=tmp_path / "processed")
    splits = builder.build_dataset(corpus, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    df_train = splits["train"]
    df_val = splits["val"]
    df_test = splits["test"]

    # Verify CSV files created
    assert (tmp_path / "processed" / "codenet_python_processed.csv").exists()
    assert (tmp_path / "processed" / "train.csv").exists()
    assert (tmp_path / "processed" / "val.csv").exists()
    assert (tmp_path / "processed" / "test.csv").exists()

    # Verify required schema
    for col in DatasetBuilder.REQUIRED_COLUMNS:
        assert col in df_train.columns
        assert col in df_val.columns
        assert col in df_test.columns

    # CRITICAL LEAKAGE TEST: Sets of problem IDs must be completely disjoint
    train_pids = set(df_train["problem_id"])
    val_pids = set(df_val["problem_id"])
    test_pids = set(df_test["problem_id"])

    assert train_pids.isdisjoint(val_pids), "CRITICAL: Train and Val contain overlapping problem IDs!"
    assert train_pids.isdisjoint(test_pids), "CRITICAL: Train and Test contain overlapping problem IDs!"
    assert val_pids.isdisjoint(test_pids), "CRITICAL: Val and Test contain overlapping problem IDs!"
