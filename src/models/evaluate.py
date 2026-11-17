"""
Comprehensive Model Evaluation & Visualization Engine
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Evaluates trained MT-ANN and Baselines on the unseen test split:
- Classification Metrics: Accuracy, Precision, Recall, Macro F1, Confusion Matrices
- Regression Metrics: MAE, MSE, RMSE, R-squared
- Generates publication-ready research figures in reports/figures/:
  1. training_loss_curves.png
  2. confusion_matrix_time.png
  3. confusion_matrix_space.png
  4. quality_parity_plot.png
  5. model_comparison_benchmark.png
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

from src.config import settings
from src.models.ann_model import MultiTaskCodeANN
from src.models.baselines import BaselineModels
from src.utils.logger import logger
from src.utils.device import get_device


class ModelEvaluator:
    """
    Evaluates multi-task models, generates comparative metrics, and exports figures.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        figures_dir: Optional[Path] = None,
        metrics_dir: Optional[Path] = None
    ):
        self.device = get_device(settings.DEVICE)
        self.model_path = model_path or (settings.CHECKPOINTS_DIR / "best_ann_model.pt")
        self.figures_dir = figures_dir or settings.FIGURES_DIR
        self.metrics_dir = metrics_dir or settings.METRICS_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.model = MultiTaskCodeANN().to(self.device)
        self._load_model()

    def _load_model(self) -> None:
        """Loads trained weights into evaluation model."""
        if self.model_path.exists():
            state_dict = torch.load(str(self.model_path), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Loaded trained MT-ANN model from {self.model_path}")
        else:
            logger.warning(f"No trained model checkpoint found at {self.model_path}")

    def evaluate_test_set(self, test_pt: Path) -> Dict[str, Any]:
        """Runs evaluation over unseen test split and exports report and figures."""
        logger.info(f"Evaluating MT-ANN on test set: {test_pt}...")
        data = torch.load(str(test_pt), weights_only=False)
        X_test = data["X"].to(self.device)
        y_time_true = data["y_time"].numpy()
        y_space_true = data["y_space"].numpy()
        y_qual_true = data["y_quality"].numpy()

        # Run inference
        with torch.no_grad():
            preds = self.model.predict(X_test)

        y_time_pred = preds["time_class_idx"]
        y_space_pred = preds["space_class_idx"]
        y_qual_pred = preds["quality_score"]

        # 1. Time Complexity Metrics
        time_acc = accuracy_score(y_time_true, y_time_pred)
        t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(y_time_true, y_time_pred, average="macro", zero_division=0)
        time_cm = confusion_matrix(y_time_true, y_time_pred)

        # 2. Space Complexity Metrics
        space_acc = accuracy_score(y_space_true, y_space_pred)
        s_prec, s_rec, s_f1, _ = precision_recall_fscore_support(y_space_true, y_space_pred, average="macro", zero_division=0)
        space_cm = confusion_matrix(y_space_true, y_space_pred)

        # 3. Quality Regression Metrics
        qual_mae = mean_absolute_error(y_qual_true, y_qual_pred)
        qual_mse = mean_squared_error(y_qual_true, y_qual_pred)
        qual_rmse = float(np.sqrt(qual_mse))
        qual_r2 = r2_score(y_qual_true, y_qual_pred)

        report = {
            "model_type": "Multi-Task ANN",
            "test_samples": len(y_time_true),
            "time_complexity": {
                "accuracy": round(float(time_acc), 4),
                "macro_precision": round(float(t_prec), 4),
                "macro_recall": round(float(t_rec), 4),
                "macro_f1": round(float(t_f1), 4),
            },
            "space_complexity": {
                "accuracy": round(float(space_acc), 4),
                "macro_precision": round(float(s_prec), 4),
                "macro_recall": round(float(s_rec), 4),
                "macro_f1": round(float(s_f1), 4),
            },
            "maintainability": {
                "mae": round(float(qual_mae), 4),
                "mse": round(float(qual_mse), 4),
                "rmse": round(float(qual_rmse), 4),
                "r2_score": round(float(qual_r2), 4),
            }
        }

        # Export Figures
        self._plot_confusion_matrix(
            time_cm,
            labels=settings.TIME_COMPLEXITY_CLASSES,
            title="Time Complexity - Confusion Matrix",
            filename="confusion_matrix_time.png"
        )
        self._plot_confusion_matrix(
            space_cm,
            labels=settings.SPACE_COMPLEXITY_CLASSES,
            title="Space Complexity - Confusion Matrix",
            filename="confusion_matrix_space.png"
        )
        self._plot_quality_parity(y_qual_true, y_qual_pred)
        self._plot_training_curves()

        # Save Metrics Report JSON
        report_path = self.metrics_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation complete. Metrics saved to {report_path}")
        return report

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        title: str,
        filename: str
    ) -> None:
        """Plots and saves normalized confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))
        # Ensure dimensions match
        display_labels = labels[:cm.shape[0]]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=display_labels,
            yticklabels=display_labels
        )
        plt.title(title, fontsize=14, pad=12)
        plt.xlabel("Predicted Class", fontsize=12)
        plt.ylabel("Ground Truth Class", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=300)
        plt.close()

    def _plot_quality_parity(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plots scatter parity plot of predicted vs true maintainability scores."""
        plt.figure(figsize=(7, 7))
        plt.scatter(y_true, y_pred, alpha=0.4, color="#2563EB", edgecolors="none")
        # 45-degree ideal diagonal
        min_v = min(np.min(y_true), np.min(y_pred))
        max_v = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_v, max_v], [min_v, max_v], "r--", label="Ideal Parity (y=x)")
        plt.title("Code Quality: Predicted vs True Maintainability", fontsize=13)
        plt.xlabel("Ground Truth Maintainability Index", fontsize=11)
        plt.ylabel("ANN Predicted Maintainability Index", fontsize=11)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "quality_parity_plot.png", dpi=300)
        plt.close()

    def _plot_training_curves(self) -> None:
        """Plots training and validation loss curves from history."""
        hist_path = self.metrics_dir / "training_history.json"
        if not hist_path.exists():
            return

        with open(hist_path, "r", encoding="utf-8") as f:
            hist = json.load(f)

        epochs = hist["epoch"]
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, hist["train_loss"], label="Train Loss", color="#3B82F6", linewidth=2)
        plt.plot(epochs, hist["val_loss"], label="Validation Loss", color="#EF4444", linewidth=2, linestyle="--")
        plt.title("Multi-Task ANN: Training & Validation Loss", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Total Loss", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "training_loss_curves.png", dpi=300)
        plt.close()

    def plot_model_comparison(
        self,
        ann_metrics: Dict[str, Any],
        rf_metrics: Dict[str, float]
    ) -> None:
        """Plots grouped bar comparison between ANN and Random Forest."""
        tasks = ["Time Acc (%)", "Space Acc (%)", "Quality R²"]
        ann_vals = [
            ann_metrics["time_complexity"]["accuracy"] * 100,
            ann_metrics["space_complexity"]["accuracy"] * 100,
            max(0, ann_metrics["maintainability"]["r2_score"] * 100),
        ]
        rf_vals = [
            rf_metrics.get("rf_time_accuracy", 0.0) * 100,
            rf_metrics.get("rf_space_accuracy", 0.0) * 100,
            max(0, rf_metrics.get("rf_quality_r2", 0.0) * 100),
        ]

        x = np.arange(len(tasks))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, ann_vals, width, label="Multi-Task ANN (Proposed)", color="#4F46E5")
        plt.bar(x + width/2, rf_vals, width, label="Random Forest Baseline", color="#9CA3AF")

        plt.ylabel("Score (%)", fontsize=12)
        plt.title("Model Comparison: MT-ANN vs Random Forest Baseline", fontsize=14)
        plt.xticks(x, tasks, fontsize=11)
        plt.ylim(0, 105)
        plt.legend(fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "model_comparison_benchmark.png", dpi=300)
        plt.close()
