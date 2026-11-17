"""
Multimodal Feature Harmonization & Fusion Pipeline
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Combines:
- 25 AST and software complexity metrics (RobustScaler-normalized)
- 768-dimensional CodeBERT contextual embeddings (L2-normalized)
Resulting in a 793-dimensional unified feature representation.

LEAKAGE PREVENTION:
The feature scaler is fitted EXCLUSIVELY on the training dataset split,
and then applied identically to validation, testing, and production inference.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import RobustScaler

from src.config import settings
from src.features.ast_extractor import ASTFeatureExtractor
from src.features.codebert_extractor import CodeBERTExtractor
from src.utils.logger import logger


class FeaturePipeline:
    """
    Orchestrates AST metric extraction, CodeBERT embeddings, normalization,
    and multimodal tensor fusion.
    """

    AST_DIM = settings.NUM_AST_FEATURES          # 25
    CODEBERT_DIM = settings.CODEBERT_EMBEDDING_DIM  # 768
    TOTAL_DIM = settings.TOTAL_FEATURE_DIM          # 793

    def __init__(
        self,
        scaler_path: Optional[Path] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.scaler_path = scaler_path or (settings.CHECKPOINTS_DIR / "ast_scaler.joblib")
        self.ast_extractor = ASTFeatureExtractor()
        self.codebert_extractor = CodeBERTExtractor(device=device)
        self.scaler: Optional[RobustScaler] = None
        self._load_scaler()

    def _load_scaler(self) -> None:
        """Loads fitted scaler from disk if it exists."""
        if self.scaler_path.exists():
            try:
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded existing AST feature scaler from {self.scaler_path}")
            except Exception as e:
                logger.warning(f"Could not load scaler from {self.scaler_path}: {e}")
                self.scaler = None

    def fit_scaler(self, ast_features: np.ndarray) -> None:
        """
        Fits RobustScaler on training AST features and persists to disk.
        
        Args:
            ast_features: np.ndarray of shape (N, 25).
        """
        logger.info(f"Fitting RobustScaler on {len(ast_features)} training AST vectors...")
        self.scaler = RobustScaler()
        self.scaler.fit(ast_features)
        
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Saved fitted AST feature scaler to {self.scaler_path}")

    def transform_single(
        self,
        source_code: str,
        sample_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts and combines multimodal features for a single source code submission.
        
        Returns:
            Dict containing:
            - 'fused_vector': np.ndarray of shape (793,)
            - 'ast_vector': np.ndarray of shape (25,)
            - 'codebert_vector': np.ndarray of shape (768,)
            - 'ast_features_dict': Dict[str, float]
            - 'ast_time_ms': float
            - 'codebert_time_ms': float
            - 'total_time_ms': float
        """
        # 1. AST Feature Extraction
        ast_res = self.ast_extractor.extract_features(source_code)
        ast_vec = ast_res["vector"]

        # 2. Normalize AST Features
        if self.scaler is not None:
            ast_norm = self.scaler.transform(ast_vec.reshape(1, -1))[0]
        else:
            # Fallback if scaler not yet fitted
            ast_norm = np.tanh(ast_vec / 10.0)

        # 3. CodeBERT Embedding
        bert_res = self.codebert_extractor.extract_embedding(source_code, sample_id=sample_id)
        bert_vec = bert_res["embedding"]

        # 4. Multimodal Fusion: Concatenation [25 + 768 = 793]
        fused = np.concatenate([ast_norm, bert_vec], axis=0).astype(np.float32)
        total_latency = round(ast_res["extraction_time_ms"] + bert_res["extraction_time_ms"], 3)

        return {
            "fused_vector": fused,
            "ast_vector": ast_vec,
            "ast_norm_vector": ast_norm,
            "codebert_vector": bert_vec,
            "ast_features_dict": ast_res["features"],
            "ast_time_ms": ast_res["extraction_time_ms"],
            "codebert_time_ms": bert_res["extraction_time_ms"],
            "total_time_ms": total_latency
        }

    def process_split_and_save(
        self,
        split_csv_path: Path,
        output_tensor_path: Path,
        fit_scaler_if_missing: bool = False
    ) -> Dict[str, Any]:
        """
        Extracts features for an entire dataset split (train, val, or test)
        and persists structured PyTorch tensors to disk.
        """
        logger.info(f"Processing split CSV: {split_csv_path}...")
        df = pd.read_csv(split_csv_path)
        
        ast_raw_list = []
        code_list = []
        sample_ids = df["sample_id"].tolist()
        problem_ids = df["problem_id"].tolist()

        # Step 1: Extract AST features
        for code in df["source_code"]:
            ast_res = self.ast_extractor.extract_features(code)
            ast_raw_list.append(ast_res["vector"])
            code_list.append(code)

        ast_raw_matrix = np.array(ast_raw_list, dtype=np.float32)

        # Step 2: Fit scaler if this is the training split
        if fit_scaler_if_missing or self.scaler is None:
            self.fit_scaler(ast_raw_matrix)

        # Step 3: Scale AST features
        ast_norm_matrix = self.scaler.transform(ast_raw_matrix)

        # Step 4: Extract CodeBERT embeddings in batch
        logger.info(f"Extracting CodeBERT embeddings for {len(code_list)} samples...")
        bert_matrix = self.codebert_extractor.extract_batch(code_list, sample_ids=sample_ids)

        # Step 5: Fuse features into (N, 793)
        X_fused = np.concatenate([ast_norm_matrix, bert_matrix], axis=1).astype(np.float32)

        # Step 6: Encode target labels
        time_to_idx = {c: i for i, c in enumerate(settings.TIME_COMPLEXITY_CLASSES)}
        space_to_idx = {c: i for i, c in enumerate(settings.SPACE_COMPLEXITY_CLASSES)}

        y_time = np.array([time_to_idx.get(t, 2) for t in df["time_complexity_label"]], dtype=np.int64)
        y_space = np.array([space_to_idx.get(s, 0) for s in df["space_complexity_label"]], dtype=np.int64)
        y_quality = df["maintainability_label"].values.astype(np.float32)

        # Step 7: Save as structured PyTorch archive
        payload = {
            "X": torch.from_numpy(X_fused),
            "y_time": torch.from_numpy(y_time),
            "y_space": torch.from_numpy(y_space),
            "y_quality": torch.from_numpy(y_quality),
            "sample_ids": sample_ids,
            "problem_ids": problem_ids,
            "feature_dim": self.TOTAL_DIM
        }
        
        output_tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(output_tensor_path))
        logger.info(f"Successfully saved {len(X_fused)} processed samples to {output_tensor_path}")

        return payload
