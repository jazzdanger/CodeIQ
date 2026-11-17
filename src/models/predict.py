"""
End-to-End Prediction & Evaluation Inference Service
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Pipeline:
Untrusted Code -> Static Sanitization -> Feature Pipeline -> MT-ANN Inference -> Diagnostic Recommendations
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from src.config import settings
from src.utils.sanitization import CodeSanitizer
from src.features.feature_pipeline import FeaturePipeline
from src.models.ann_model import MultiTaskCodeANN
from src.utils.logger import logger
from src.utils.device import get_device


class CodePredictor:
    """
    Production inference engine for static code grading, complexity prediction,
    and automated architectural recommendations.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        feature_pipeline: Optional[FeaturePipeline] = None
    ):
        self.device = get_device(settings.DEVICE)
        self.pipeline = feature_pipeline or FeaturePipeline()
        self.model_path = model_path or (settings.CHECKPOINTS_DIR / "best_ann_model.pt")
        self.model = MultiTaskCodeANN().to(self.device)
        self._load_weights()

    def _load_weights(self) -> None:
        """Loads trained weights if available."""
        if self.model_path.exists():
            state_dict = torch.load(str(self.model_path), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Loaded trained MT-ANN weights from {self.model_path}")
        else:
            logger.warning(f"No trained model checkpoint found at {self.model_path}. Using initial weights.")

    def analyze_code(self, source_code: str, language: str = "python") -> Dict[str, Any]:
        """
        Executes complete static analysis pipeline on untrusted code.
        
        Returns:
            Dict containing:
            - 'is_valid': bool
            - 'error': Optional[str]
            - 'time_complexity': Dict (label, confidence, probabilities)
            - 'space_complexity': Dict (label, confidence, probabilities)
            - 'code_quality': Dict (overall_score, maintainability_index, cyclomatic_complexity)
            - 'ast_metrics': Dict[str, float]
            - 'recommendations': List[str]
            - 'latency_breakdown_ms': Dict[str, float]
        """
        # Step 1: Input Validation & Sanitization (Strict Non-Executing)
        is_valid, err_msg, meta = CodeSanitizer.validate_submission(source_code, language)
        if not is_valid:
            return {
                "success": False,
                "error": err_msg,
                "metadata": meta
            }

        # Step 2: Multimodal Feature Extraction
        feat_data = self.pipeline.transform_single(source_code)
        fused_tensor = torch.from_numpy(feat_data["fused_vector"]).unsqueeze(0).to(self.device)

        # Step 3: Model Inference
        self.model.eval()
        with torch.no_grad():
            preds = self.model.predict(fused_tensor)

        time_idx = int(preds["time_class_idx"][0])
        space_idx = int(preds["space_class_idx"][0])
        time_label = settings.TIME_COMPLEXITY_CLASSES[time_idx]
        space_label = settings.SPACE_COMPLEXITY_CLASSES[space_idx]
        time_conf = float(preds["time_confidence"][0])
        space_conf = float(preds["space_confidence"][0])
        quality_score = float(preds["quality_score"][0])

        # Step 4: Build Diagnostic Recommendations
        ast_dict = feat_data["ast_features_dict"]
        recommendations = self._generate_recommendations(ast_dict, time_label, space_label, quality_score)

        return {
            "success": True,
            "error": None,
            "time_complexity": {
                "predicted_class": time_label,
                "confidence": round(time_conf * 100.0, 2),
                "distribution": {
                    cls_name: round(float(p) * 100.0, 2)
                    for cls_name, p in zip(settings.TIME_COMPLEXITY_CLASSES, preds["time_probs"][0])
                }
            },
            "space_complexity": {
                "predicted_class": space_label,
                "confidence": round(space_conf * 100.0, 2),
                "distribution": {
                    cls_name: round(float(p) * 100.0, 2)
                    for cls_name, p in zip(settings.SPACE_COMPLEXITY_CLASSES, preds["space_probs"][0])
                }
            },
            "code_quality": {
                "overall_score": round(max(0.0, min(100.0, quality_score)), 1),
                "cyclomatic_complexity": ast_dict.get("cyclomatic_complexity", 1.0),
                "max_nesting_depth": ast_dict.get("max_loop_nesting_depth", 0.0),
                "halstead_difficulty": ast_dict.get("halstead_difficulty", 0.0),
            },
            "ast_metrics": ast_dict,
            "recommendations": recommendations,
            "latency_breakdown_ms": {
                "ast_extraction_ms": feat_data["ast_time_ms"],
                "codebert_extraction_ms": feat_data["codebert_time_ms"],
                "total_pipeline_ms": feat_data["total_time_ms"],
            }
        }

    @staticmethod
    def _generate_recommendations(
        ast_dict: Dict[str, float],
        time_label: str,
        space_label: str,
        quality_score: float
    ) -> List[str]:
        """Generates actionable refactoring suggestions based on AST metrics."""
        recs = []

        nesting = ast_dict.get("max_loop_nesting_depth", 0.0)
        if nesting >= 3.0:
            recs.append(f"Deep loop nesting detected ({int(nesting)} levels). Consider refactoring inner loops using hash tables or vector operations.")
        elif nesting == 2.0 and time_label in ("O(N^2)", "O(N^3)"):
            recs.append("Quadratic nested iteration detected. Check if sorting + two pointers or sliding window can reduce complexity.")

        cc = ast_dict.get("cyclomatic_complexity", 1.0)
        if cc > 10.0:
            recs.append(f"High Cyclomatic Complexity ({int(cc)}). Decompose large monolithic conditionals into smaller modular helper functions.")

        rec_fn = ast_dict.get("num_recursive_functions", 0.0)
        if rec_fn > 0 and time_label in ("O(2^N)", "O(N^3)"):
            recs.append("Recursive branching without observed memoization. Consider applying @functools.lru_cache or dynamic programming.")

        reuse = ast_dict.get("variable_reuse_ratio", 1.0)
        if reuse < 1.5:
            recs.append("Low variable reuse ratio. Multiple transient variables created with single usages.")

        if quality_score < 65.0:
            recs.append("Overall maintainability score is below recommended threshold. Refactor nested logic and simplify branching conditions.")

        if not recs:
            recs.append("Clean algorithmic structure. Efficient control flow and maintainable structural metrics.")

        return recs
