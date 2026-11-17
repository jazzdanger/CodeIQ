"""
Standalone Model Evaluation & Visualization Entrypoint
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Usage:
    python evaluate.py
    or
    .\venv\Scripts\python evaluate.py
"""

import os
import sys
from pathlib import Path

# Enforce clean PyTorch-only mode and local Matplotlib cache automatically
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

project_root = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(project_root / "reports" / "figures" / ".mpl")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import settings
from src.models.evaluate import ModelEvaluator

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("EVALUATING BEST MODEL ON TEST SPLIT (1,400 SAMPLES)")
    print("=" * 65 + "\n")
    test_pt = settings.PROCESSED_DATA_DIR / "test_features.pt"
    evaluator = ModelEvaluator()
    report = evaluator.evaluate_test_set(test_pt)
    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"Report saved to: {settings.METRICS_DIR / 'evaluation_report.json'}")
    print(f"Figures saved to: {settings.FIGURES_DIR}")
    print("=" * 65 + "\n")
