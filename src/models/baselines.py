"""
Baseline Machine Learning Models Module
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Implements competitive ML baselines for research evaluation:
1. Random Forest Classifier & Regressor (Non-linear ensemble)
2. Logistic Regression & Ridge Regressor (Linear baseline)

All baselines are evaluated on the exact same 793-dimensional precomputed multimodal features.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

from src.config import settings
from src.utils.logger import logger


class BaselineModels:
    """
    Trains, evaluates, and persists standard ML baselines for academic benchmark comparison.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.BASELINES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train_time: np.ndarray,
        y_train_space: np.ndarray,
        y_train_qual: np.ndarray,
        n_estimators: int = 100,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Trains Random Forest models across all three tasks."""
        logger.info(f"Training Random Forest baselines (n_estimators={n_estimators})...")
        
        # 1. Time Complexity Classifier
        rf_time = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        rf_time.fit(X_train, y_train_time)

        # 2. Space Complexity Classifier
        rf_space = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        rf_space.fit(X_train, y_train_space)

        # 3. Quality Regressor
        rf_qual = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        rf_qual.fit(X_train, y_train_qual)

        # Save artifacts
        joblib.dump(rf_time, self.output_dir / "rf_time.joblib")
        joblib.dump(rf_space, self.output_dir / "rf_space.joblib")
        joblib.dump(rf_qual, self.output_dir / "rf_quality.joblib")
        logger.info(f"Random Forest models saved to {self.output_dir}")

        return {
            "rf_time": rf_time,
            "rf_space": rf_space,
            "rf_quality": rf_qual
        }

    def train_linear_baselines(
        self,
        X_train: np.ndarray,
        y_train_time: np.ndarray,
        y_train_space: np.ndarray,
        y_train_qual: np.ndarray,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Trains Logistic Regression and Ridge Regressor baselines."""
        logger.info("Training Linear / Logistic baselines...")

        lr_time = LogisticRegression(max_iter=1000, random_state=seed)
        lr_time.fit(X_train, y_train_time)

        lr_space = LogisticRegression(max_iter=1000, random_state=seed)
        lr_space.fit(X_train, y_train_space)

        ridge_qual = Ridge(alpha=1.0, random_state=seed)
        ridge_qual.fit(X_train, y_train_qual)

        joblib.dump(lr_time, self.output_dir / "lr_time.joblib")
        joblib.dump(lr_space, self.output_dir / "lr_space.joblib")
        joblib.dump(ridge_qual, self.output_dir / "ridge_quality.joblib")

        return {
            "lr_time": lr_time,
            "lr_space": lr_space,
            "lr_quality": ridge_qual
        }

    def evaluate_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test_time: np.ndarray,
        y_test_space: np.ndarray,
        y_test_qual: np.ndarray,
        prefix: str = "rf"
    ) -> Dict[str, float]:
        """Evaluates trained baseline models on unseen test split."""
        model_time = models[f"{prefix}_time"]
        model_space = models[f"{prefix}_space"]
        model_qual = models[f"{prefix}_quality"]

        pred_time = model_time.predict(X_test)
        pred_space = model_space.predict(X_test)
        pred_qual = model_qual.predict(X_test)

        time_acc = accuracy_score(y_test_time, pred_time)
        time_f1 = f1_score(y_test_time, pred_time, average="macro", zero_division=0)

        space_acc = accuracy_score(y_test_space, pred_space)
        space_f1 = f1_score(y_test_space, pred_space, average="macro", zero_division=0)

        qual_mae = mean_absolute_error(y_test_qual, pred_qual)
        qual_r2 = r2_score(y_test_qual, pred_qual)

        metrics = {
            f"{prefix}_time_accuracy": round(float(time_acc), 4),
            f"{prefix}_time_macro_f1": round(float(time_f1), 4),
            f"{prefix}_space_accuracy": round(float(space_acc), 4),
            f"{prefix}_space_macro_f1": round(float(space_f1), 4),
            f"{prefix}_quality_mae": round(float(qual_mae), 4),
            f"{prefix}_quality_r2": round(float(qual_r2), 4),
        }
        logger.info(f"{prefix.upper()} Benchmark Results: Time Acc={time_acc:.4f}, Space Acc={space_acc:.4f}, Quality MAE={qual_mae:.4f}")
        return metrics
