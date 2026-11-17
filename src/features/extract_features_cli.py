"""
Phase 3 Feature Extraction & Multimodal Precomputation CLI
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Executes Phase 3 end-to-end:
1. Extracts 25 structural AST features for train, val, and test splits.
2. Fits RobustScaler EXCLUSIVELY on training AST features and persists to disk.
3. Generates 768-dimensional normalized CodeBERT contextual embeddings.
4. Fuses features into unified 793-dimensional multimodal tensors.
5. Saves processed PyTorch tensor datasets (train_features.pt, val_features.pt, test_features.pt).
6. Records feature extraction latency metrics.
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.features.feature_pipeline import FeaturePipeline
from src.utils.logger import logger


def run_feature_extraction(max_samples_per_split: Optional[int] = None):
    logger.info("=" * 60)
    logger.info("STARTING PHASE 3: FEATURE EXTRACTION & FUSION PIPELINE")
    logger.info("=" * 60)

    start_total = time.perf_counter()
    pipeline = FeaturePipeline()

    train_csv = settings.PROCESSED_DATA_DIR / "train.csv"
    val_csv = settings.PROCESSED_DATA_DIR / "val.csv"
    test_csv = settings.PROCESSED_DATA_DIR / "test.csv"

    train_pt = settings.PROCESSED_DATA_DIR / "train_features.pt"
    val_pt = settings.PROCESSED_DATA_DIR / "val_features.pt"
    test_pt = settings.PROCESSED_DATA_DIR / "test_features.pt"

    # 1. Process Train Split & Fit Scaler
    logger.info("Step 1/3: Processing Training Split (Fitting RobustScaler)...")
    train_data = pipeline.process_split_and_save(train_csv, train_pt, fit_scaler_if_missing=True)

    # 2. Process Validation Split
    logger.info("Step 2/3: Processing Validation Split...")
    val_data = pipeline.process_split_and_save(val_csv, val_pt, fit_scaler_if_missing=False)

    # 3. Process Test Split
    logger.info("Step 3/3: Processing Test Split...")
    test_data = pipeline.process_split_and_save(test_csv, test_pt, fit_scaler_if_missing=False)

    total_duration_sec = round(time.perf_counter() - start_total, 2)

    # Compile Summary
    summary = {
        "status": "success",
        "num_ast_features": pipeline.AST_DIM,
        "codebert_embedding_dim": pipeline.CODEBERT_DIM,
        "total_multimodal_dim": pipeline.TOTAL_DIM,
        "train_samples": int(len(train_data["X"])),
        "val_samples": int(len(val_data["X"])),
        "test_samples": int(len(test_data["X"])),
        "total_processed_samples": int(len(train_data["X"]) + len(val_data["X"]) + len(test_data["X"])),
        "tensor_shapes": {
            "train_X": list(train_data["X"].shape),
            "val_X": list(val_data["X"].shape),
            "test_X": list(test_data["X"].shape),
        },
        "scaler_artifact": str(pipeline.scaler_path),
        "total_extraction_duration_seconds": total_duration_sec
    }

    metrics_file = settings.METRICS_DIR / "feature_extraction_summary.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("PHASE 3 FEATURE EXTRACTION COMPLETE:")
    logger.info(f"- Train Tensors Shape : {train_data['X'].shape}")
    logger.info(f"- Val Tensors Shape   : {val_data['X'].shape}")
    logger.info(f"- Test Tensors Shape  : {test_data['X'].shape}")
    logger.info(f"- Total Feature Dim   : {pipeline.TOTAL_DIM} (25 AST + 768 CodeBERT)")
    logger.info(f"- Elapsed Duration    : {total_duration_sec}s")
    logger.info(f"- Summary Written To  : {metrics_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_feature_extraction()
