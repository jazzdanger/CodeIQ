"""
Phase 4 Master Model Training & Academic Benchmarking CLI
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Executes the complete Phase 4 machine learning workflow:
1. Trains the Multi-Task Artificial Neural Network (MT-ANN) on CUDA GPU.
2. Applies Early Stopping (patience=8) and saves the best checkpoint.
3. Trains Baseline Models (Random Forest, Logistic Regression, Ridge).
4. Evaluates all models on the UNSEEN test split (1,400 samples).
5. Generates publication-ready figures in reports/figures/:
   - training_loss_curves.png
   - confusion_matrix_time.png
   - confusion_matrix_space.png
   - quality_parity_plot.png
   - model_comparison_benchmark.png
6. Exports comprehensive benchmark metrics report to reports/metrics/evaluation_report.json.
"""

import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.config import settings
from src.models.ann_model import MultiTaskCodeANN
from src.models.train import MultiTaskTrainer
from src.models.baselines import BaselineModels
from src.models.evaluate import ModelEvaluator
from src.utils.logger import logger
from src.utils.device import get_device


def run_training_pipeline(max_epochs: int = 35):
    logger.info("=" * 65)
    logger.info("STARTING PHASE 4: MULTI-TASK ANN TRAINING & BENCHMARKING")
    logger.info("=" * 65)

    device = get_device(settings.DEVICE)
    logger.info(f"Target Compute Device: {device}")

    train_pt = settings.PROCESSED_DATA_DIR / "train_features.pt"
    val_pt = settings.PROCESSED_DATA_DIR / "val_features.pt"
    test_pt = settings.PROCESSED_DATA_DIR / "test_features.pt"

    # -------------------------------------------------------------
    # 1. Train Proposed Multi-Task ANN
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 1: Training Proposed Multi-Task ANN...")
    trainer = MultiTaskTrainer(device=device, batch_size=settings.TRAINING_BATCH_SIZE)
    history = trainer.train(
        train_pt=train_pt,
        val_pt=val_pt,
        max_epochs=max_epochs,
        patience=8,
        seed=settings.SEED
    )

    # -------------------------------------------------------------
    # 2. Train Random Forest & Linear Baselines
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 2: Training Academic Baseline Models (Random Forest & Linear)...")
    train_data = torch.load(str(train_pt), weights_only=False)
    X_train_np = train_data["X"].numpy()
    y_time_train = train_data["y_time"].numpy()
    y_space_train = train_data["y_space"].numpy()
    y_qual_train = train_data["y_quality"].numpy()

    test_data = torch.load(str(test_pt), weights_only=False)
    X_test_np = test_data["X"].numpy()
    y_time_test = test_data["y_time"].numpy()
    y_space_test = test_data["y_space"].numpy()
    y_qual_test = test_data["y_quality"].numpy()

    baselines = BaselineModels()
    rf_models = baselines.train_random_forest(
        X_train_np, y_time_train, y_space_train, y_qual_train, n_estimators=80, seed=settings.SEED
    )
    rf_metrics = baselines.evaluate_models(
        rf_models, X_test_np, y_time_test, y_space_test, y_qual_test, prefix="rf"
    )

    linear_models = baselines.train_linear_baselines(
        X_train_np, y_time_train, y_space_train, y_qual_train, seed=settings.SEED
    )
    linear_metrics = baselines.evaluate_models(
        linear_models, X_test_np, y_time_test, y_space_test, y_qual_test, prefix="lr"
    )

    # -------------------------------------------------------------
    # 3. Comprehensive Evaluation of MT-ANN on Test Set
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 3: Evaluating Best MT-ANN on Unseen Test Set (1,400 samples)...")
    evaluator = ModelEvaluator()
    ann_report = evaluator.evaluate_test_set(test_pt)

    # Plot Model Comparison Bar Chart
    evaluator.plot_model_comparison(ann_report, rf_metrics)

    # Compile Final Comparative Benchmark JSON
    final_benchmark = {
        "dataset_splits": {
            "train_samples": len(X_train_np),
            "val_samples": len(torch.load(str(val_pt), weights_only=False)["X"]),
            "test_samples": len(X_test_np)
        },
        "proposed_mt_ann": ann_report,
        "baseline_random_forest": rf_metrics,
        "baseline_linear_models": linear_metrics,
        "figures_generated": [
            "training_loss_curves.png",
            "confusion_matrix_time.png",
            "confusion_matrix_space.png",
            "quality_parity_plot.png",
            "model_comparison_benchmark.png"
        ]
    }

    benchmark_path = settings.METRICS_DIR / "final_model_benchmark.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(final_benchmark, f, indent=2)

    logger.info("=" * 65)
    logger.info("PHASE 4 TRAINING & BENCHMARKING COMPLETE!")
    logger.info("=" * 65)
    logger.info("Comparative Test Set Summary:")
    logger.info(f"  MT-ANN Time Accuracy   : {ann_report['time_complexity']['accuracy']*100:.2f}% (Macro F1: {ann_report['time_complexity']['macro_f1']:.4f})")
    logger.info(f"  RF Baseline Time Acc   : {rf_metrics['rf_time_accuracy']*100:.2f}% (Macro F1: {rf_metrics['rf_time_macro_f1']:.4f})")
    logger.info(f"  MT-ANN Space Accuracy  : {ann_report['space_complexity']['accuracy']*100:.2f}% (Macro F1: {ann_report['space_complexity']['macro_f1']:.4f})")
    logger.info(f"  RF Baseline Space Acc  : {rf_metrics['rf_space_accuracy']*100:.2f}% (Macro F1: {rf_metrics['rf_space_macro_f1']:.4f})")
    logger.info(f"  MT-ANN Quality MAE     : {ann_report['maintainability']['mae']:.2f} (R²: {ann_report['maintainability']['r2_score']:.4f})")
    logger.info(f"  RF Quality MAE         : {rf_metrics['rf_quality_mae']:.2f} (R²: {rf_metrics['rf_quality_r2']:.4f})")
    logger.info("=" * 65)
    logger.info(f"Generated figures in: {settings.FIGURES_DIR}")
    logger.info(f"Final benchmark saved to: {benchmark_path}")
    logger.info("=" * 65)


if __name__ == "__main__":
    run_training_pipeline()
