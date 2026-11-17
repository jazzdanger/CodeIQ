"""
Multi-Task ANN Training Module
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Features:
- Joint multi-task optimization for Time Complexity, Space Complexity, and Code Quality
- Dynamic multi-task loss weighting
- AdamW optimizer with ReduceLROnPlateau learning rate scheduler
- Early stopping tracking validation loss
- Model checkpointing (best_ann_model.pt)
- Full epoch metric logging and history persistence (training_history.json)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from src.config import settings
from src.models.ann_model import MultiTaskCodeANN
from src.utils.logger import logger
from src.utils.device import get_device, set_seed


class MultiTaskTrainer:
    """
    Manages end-to-end training and checkpointing of the MultiTaskCodeANN.
    """

    def __init__(
        self,
        model: Optional[MultiTaskCodeANN] = None,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = settings.TRAINING_BATCH_SIZE,
        checkpoint_dir: Optional[Path] = None
    ):
        self.device = device or get_device(settings.DEVICE)
        self.model = (model or MultiTaskCodeANN()).to(self.device)
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir or settings.CHECKPOINTS_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / "best_ann_model.pt"

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3
        )

        # Loss Functions
        self.loss_time_fn = nn.CrossEntropyLoss()
        self.loss_space_fn = nn.CrossEntropyLoss()
        self.loss_quality_fn = nn.SmoothL1Loss()  # Huber loss (robust to outliers)

        # Loss Weights
        self.weight_time = 1.0
        self.weight_space = 0.5
        self.weight_quality = 0.05  # Scaled to balance with cross-entropy magnitudes

    def load_dataset_split(self, pt_path: Path) -> TensorDataset:
        """Loads precomputed feature tensor payload into PyTorch TensorDataset."""
        data = torch.load(str(pt_path), weights_only=False)
        return TensorDataset(
            data["X"],
            data["y_time"],
            data["y_space"],
            data["y_quality"]
        )

    def train(
        self,
        train_pt: Path,
        val_pt: Path,
        max_epochs: int = 40,
        patience: int = 8,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Executes training loop with early stopping, validation monitoring, and checkpointing.
        """
        set_seed(seed)
        logger.info(f"Loading training data from {train_pt}...")
        train_dataset = self.load_dataset_split(train_pt)
        val_dataset = self.load_dataset_split(val_pt)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        logger.info(f"Loaded {len(train_dataset)} train samples, {len(val_dataset)} val samples.")
        logger.info(f"Starting Multi-Task ANN training on {self.device} (Max Epochs: {max_epochs})...")

        best_val_loss = float("inf")
        epochs_no_improve = 0
        history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_time_acc": [],
            "val_space_acc": [],
            "val_quality_mae": [],
            "learning_rate": []
        }

        start_time = time.perf_counter()

        for epoch in range(1, max_epochs + 1):
            # --- TRAINING PHASE ---
            self.model.train()
            train_loss_accum = 0.0

            for X_b, y_time_b, y_space_b, y_qual_b in train_loader:
                X_b = X_b.to(self.device)
                y_time_b = y_time_b.to(self.device)
                y_space_b = y_space_b.to(self.device)
                y_qual_b = y_qual_b.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_b)

                # Compute task losses
                l_time = self.loss_time_fn(outputs["time_logits"], y_time_b)
                l_space = self.loss_space_fn(outputs["space_logits"], y_space_b)
                l_qual = self.loss_quality_fn(outputs["quality_score"].squeeze(-1), y_qual_b)

                total_loss = (
                    self.weight_time * l_time +
                    self.weight_space * l_space +
                    self.weight_quality * l_qual
                )

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss_accum += total_loss.item() * len(X_b)

            train_loss = train_loss_accum / len(train_dataset)

            # --- VALIDATION PHASE ---
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["val_loss"]

            curr_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            history["epoch"].append(epoch)
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_time_acc"].append(round(val_metrics["time_acc"], 4))
            history["val_space_acc"].append(round(val_metrics["space_acc"], 4))
            history["val_quality_mae"].append(round(val_metrics["quality_mae"], 4))
            history["learning_rate"].append(curr_lr)

            logger.info(
                f"Epoch {epoch:02d}/{max_epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Time Acc: {val_metrics['time_acc']*100:.1f}% | "
                f"Space Acc: {val_metrics['space_acc']*100:.1f}% | "
                f"Qual MAE: {val_metrics['quality_mae']:.2f} | "
                f"LR: {curr_lr:.2e}"
            )

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), str(self.best_model_path))
                logger.info(f"  --> Saved new best model checkpoint (Val Loss: {val_loss:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (Patience: {patience}).")
                    break

        elapsed = round(time.perf_counter() - start_time, 2)
        logger.info(f"Training completed in {elapsed}s. Best Val Loss: {best_val_loss:.4f}")

        # Save training history
        history_path = settings.METRICS_DIR / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return history

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluates model performance on given DataLoader."""
        self.model.eval()
        total_loss = 0.0
        all_time_preds, all_time_true = [], []
        all_space_preds, all_space_true = [], []
        all_qual_preds, all_qual_true = [], []

        with torch.no_grad():
            for X_b, y_time_b, y_space_b, y_qual_b in loader:
                X_b = X_b.to(self.device)
                y_time_b = y_time_b.to(self.device)
                y_space_b = y_space_b.to(self.device)
                y_qual_b = y_qual_b.to(self.device)

                outputs = self.model(X_b)

                l_time = self.loss_time_fn(outputs["time_logits"], y_time_b)
                l_space = self.loss_space_fn(outputs["space_logits"], y_space_b)
                l_qual = self.loss_quality_fn(outputs["quality_score"].squeeze(-1), y_qual_b)

                loss = (
                    self.weight_time * l_time +
                    self.weight_space * l_space +
                    self.weight_quality * l_qual
                )
                total_loss += loss.item() * len(X_b)

                time_preds = torch.argmax(outputs["time_logits"], dim=-1).cpu().numpy()
                space_preds = torch.argmax(outputs["space_logits"], dim=-1).cpu().numpy()
                qual_preds = outputs["quality_score"].squeeze(-1).cpu().numpy()

                all_time_preds.extend(time_preds)
                all_time_true.extend(y_time_b.cpu().numpy())
                all_space_preds.extend(space_preds)
                all_space_true.extend(y_space_b.cpu().numpy())
                all_qual_preds.extend(qual_preds)
                all_qual_true.extend(y_qual_b.cpu().numpy())

        total_samples = len(all_time_true)
        val_loss = total_loss / total_samples
        time_acc = accuracy_score(all_time_true, all_time_preds)
        space_acc = accuracy_score(all_space_true, all_space_preds)
        qual_mae = mean_absolute_error(all_qual_true, all_qual_preds)

        return {
            "val_loss": val_loss,
            "time_acc": time_acc,
            "space_acc": space_acc,
            "quality_mae": qual_mae
        }
