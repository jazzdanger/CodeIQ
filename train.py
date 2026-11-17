"""
Standalone Model Training & Evaluation Entrypoint
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Usage:
    python train.py
    or
    .\venv\Scripts\python train.py
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

from src.models.train_and_evaluate_cli import run_training_pipeline

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("STARTING MODEL TRAINING & EVALUATION (AI CODE GRADER)")
    print("=" * 65 + "\n")
    run_training_pipeline()
