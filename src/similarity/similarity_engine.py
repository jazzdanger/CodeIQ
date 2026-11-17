"""
Multi-Criteria Code Similarity Engine
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Calculates three-tiered code similarity:
1. Semantic Similarity: Cosine similarity of 768-d CodeBERT embeddings.
2. Structural Similarity: Cosine similarity of 25 normalized AST feature vectors.
3. Lexical Similarity: Jaccard similarity of tokenized code n-grams.

ETHICAL DISCLAIMER:
Similarity metrics measure syntactic and semantic overlap.
High similarity serves as evidence for human instructor review, NOT proof of plagiarism.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from src.features.feature_pipeline import FeaturePipeline
from src.utils.logger import logger


class CodeSimilarityEngine:
    """
    Computes semantic, structural, and lexical similarity between code submissions.
    """

    DISCLAIMER = (
        "ACADEMIC NOTICE: Similarity scores represent structural, semantic, and token-level "
        "overlap. Similarity is evidence for human instructor review, NOT proof of academic misconduct."
    )

    def __init__(self, feature_pipeline: Optional[FeaturePipeline] = None):
        self.pipeline = feature_pipeline or FeaturePipeline()

    def compare_submissions(
        self,
        code_a: str,
        code_b: str,
        name_a: str = "Submission A",
        name_b: str = "Submission B"
    ) -> Dict[str, Any]:
        """
        Performs three-tiered similarity comparison between two code submissions.
        
        Returns:
            Dict containing:
            - 'composite_similarity': float (0.0 to 100.0)
            - 'semantic_similarity': float (0.0 to 100.0)
            - 'structural_similarity': float (0.0 to 100.0)
            - 'lexical_similarity': float (0.0 to 100.0)
            - 'risk_level': str ('Low', 'Moderate', 'High', 'Critical')
            - 'disclaimer': str
        """
        # 1. Extract Multimodal Features
        feat_a = self.pipeline.transform_single(code_a)
        feat_b = self.pipeline.transform_single(code_b)

        # 2. Semantic Similarity (CodeBERT Cosine Similarity)
        emb_a = feat_a["codebert_vector"]
        emb_b = feat_b["codebert_vector"]
        semantic_sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9))
        semantic_score = max(0.0, min(100.0, semantic_sim * 100.0))

        # 3. Structural Similarity (AST Metric Vector Cosine)
        ast_a = feat_a["ast_norm_vector"]
        ast_b = feat_b["ast_norm_vector"]
        norm_product = np.linalg.norm(ast_a) * np.linalg.norm(ast_b) + 1e-9
        structural_sim = float(np.dot(ast_a, ast_b) / norm_product)
        structural_score = max(0.0, min(100.0, structural_sim * 100.0))

        # 4. Lexical Token Jaccard Similarity
        lexical_sim = self._compute_token_jaccard(code_a, code_b)
        lexical_score = max(0.0, min(100.0, lexical_sim * 100.0))

        # 5. Composite Weighted Similarity
        # Weighting: 50% Semantic (CodeBERT) + 30% Structural (AST) + 20% Lexical (Tokens)
        composite = (0.50 * semantic_score) + (0.30 * structural_score) + (0.20 * lexical_score)
        composite_score = round(max(0.0, min(100.0, composite)), 2)

        # 6. Risk Level
        if composite_score >= 85.0:
            risk = "Critical (High Overlap)"
        elif composite_score >= 70.0:
            risk = "High"
        elif composite_score >= 45.0:
            risk = "Moderate"
        else:
            risk = "Low"

        return {
            "submission_a": name_a,
            "submission_b": name_b,
            "composite_similarity": composite_score,
            "semantic_similarity": round(semantic_score, 2),
            "structural_similarity": round(structural_score, 2),
            "lexical_similarity": round(lexical_score, 2),
            "risk_level": risk,
            "disclaimer": self.DISCLAIMER,
        }

    @staticmethod
    def _compute_token_jaccard(code_a: str, code_b: str) -> float:
        """Computes Jaccard similarity of normalized alphanumeric code tokens."""
        tokens_a = set(re.findall(r"\b\w+\b", code_a.lower()))
        tokens_b = set(re.findall(r"\b\w+\b", code_b.lower()))

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return float(intersection / union)
