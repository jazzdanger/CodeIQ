"""
Multi-Task Artificial Neural Network (MT-ANN) Architecture
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Architecture:
Input: 793 dimensions (25 normalized AST features + 768 CodeBERT embeddings)

Shared Representation Backbone:
- Linear(793 -> 384) + LayerNorm + Mish activation + Dropout(0.2)
- Residual Block: Linear(384 -> 384) + LayerNorm + Mish + Dropout(0.2) with identity skip
- Projection: Linear(384 -> 192) + LayerNorm + Mish + Dropout(0.1)

Specialized Multi-Task Output Heads:
- Head 1 (Time Complexity): Linear(192 -> 7) [O(1), O(log N), O(N), O(N log N), O(N^2), O(N^3), O(2^N)]
- Head 2 (Space Complexity): Linear(192 -> 4) [O(1), O(log N), O(N), O(N^2)]
- Head 3 (Code Quality / Maintainability): Linear(192 -> 64) -> Mish -> Linear(64 -> 1) -> Sigmoid() * 100.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from src.config import settings


class ResidualBlock(nn.Module):
    """
    Bottleneck residual block with LayerNorm, Mish activation, and dropout.
    """
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class MultiTaskCodeANN(nn.Module):
    """
    Multi-Task Artificial Neural Network for joint prediction of:
    1. Time Complexity (7-class classification)
    2. Space Complexity (4-class classification)
    3. Maintainability / Code Quality (0 - 100 continuous regression)
    """

    def __init__(
        self,
        input_dim: int = settings.TOTAL_FEATURE_DIM,       # 793
        num_time_classes: int = len(settings.TIME_COMPLEXITY_CLASSES),   # 7
        num_space_classes: int = len(settings.SPACE_COMPLEXITY_CLASSES), # 4
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_time_classes = num_time_classes
        self.num_space_classes = num_space_classes

        # Shared Backbone Encoder
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.LayerNorm(384),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )

        self.res_block1 = ResidualBlock(384, dropout_rate)
        
        self.bottleneck = nn.Sequential(
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.Mish(),
            nn.Dropout(dropout_rate * 0.5),
        )

        # Output Head 1: Time Complexity Classification (7 classes)
        self.head_time = nn.Sequential(
            nn.Linear(192, 96),
            nn.Mish(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(96, num_time_classes)
        )

        # Output Head 2: Space Complexity Classification (4 classes)
        self.head_space = nn.Sequential(
            nn.Linear(192, 64),
            nn.Mish(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_space_classes)
        )

        # Output Head 3: Code Quality & Maintainability Regression (0 to 100)
        self.head_quality = nn.Sequential(
            nn.Linear(192, 64),
            nn.Mish(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Maps to [0, 1], then scaled by 100.0 in forward
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared representation and all 3 task heads.
        
        Args:
            x: Input tensor of shape (batch_size, 793).
            
        Returns:
            Dict containing:
            - 'time_logits': Tensor of shape (batch_size, 7)
            - 'space_logits': Tensor of shape (batch_size, 4)
            - 'quality_score': Tensor of shape (batch_size, 1) in range [0.0, 100.0]
            - 'shared_representation': Tensor of shape (batch_size, 192)
        """
        h = self.input_proj(x)
        h = self.res_block1(h)
        shared_feat = self.bottleneck(h)

        time_logits = self.head_time(shared_feat)
        space_logits = self.head_space(shared_feat)
        # Scale Sigmoid output [0, 1] to [0.0, 100.0] maintainability scale
        quality_score = self.head_quality(shared_feat) * 100.0

        return {
            "time_logits": time_logits,
            "space_logits": space_logits,
            "quality_score": quality_score,
            "shared_representation": shared_feat
        }

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Inference helper with softmax probabilities and decoded class labels.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            time_probs = F.softmax(outputs["time_logits"], dim=-1)
            space_probs = F.softmax(outputs["space_logits"], dim=-1)

            time_preds = torch.argmax(time_probs, dim=-1)
            space_preds = torch.argmax(space_probs, dim=-1)

            return {
                "time_class_idx": time_preds.cpu().numpy(),
                "time_confidence": torch.max(time_probs, dim=-1).values.cpu().numpy(),
                "time_probs": time_probs.cpu().numpy(),
                "space_class_idx": space_preds.cpu().numpy(),
                "space_confidence": torch.max(space_probs, dim=-1).values.cpu().numpy(),
                "space_probs": space_probs.cpu().numpy(),
                "quality_score": outputs["quality_score"].squeeze(-1).cpu().numpy(),
            }
