"""
Ground-Truth Label Generation Module
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Annotation & Labeling Methodology:
1. Time Complexity: Asymptotic Big-O classes assigned from verified algorithmic taxonomy
   [O(1), O(log N), O(N), O(N log N), O(N^2), O(N^3), O(2^N)].
2. Space Complexity: Auxiliary memory classification [O(1), O(log N), O(N), O(N^2)].
3. Maintainability: Ground-truth continuous score [0, 100] computed via Software Engineering
   standard Radon Maintainability Index (MI), formulated from Halstead Volume,
   McCabe Cyclomatic Complexity, and Lines of Code:
   MI = max(0.0, min(100.0, (171 - 5.2*ln(V) - 0.23*CC - 16.2*ln(LOC)) * 100 / 171))
4. Code Similarity Group: Problem ID partition ensuring semantic equivalence ground truth.
"""

import math
from typing import Dict, Any, Optional
import radon.metrics
from radon.complexity import cc_visit
from src.config import settings
from src.data.taxonomy import get_problem_spec, PROBLEM_REGISTRY
from src.utils.logger import logger


class LabelGenerator:
    """
    Computes rigorous ground-truth multi-task target labels for code submissions.
    """

    @classmethod
    def compute_maintainability_index(cls, source_code: str) -> float:
        """
        Computes the software engineering standard Maintainability Index (MI)
        in the range [0.0, 100.0] using Radon.
        
        Args:
            source_code: Validated Python source code string.
            
        Returns:
            Continuous maintainability score between 0.0 and 100.0.
        """
        try:
            # radon.metrics.mi_visit computes standard MI in range [0, 100]
            raw_mi = radon.metrics.mi_visit(source_code, multi=False)
            # Bound strictly between 0.0 and 100.0
            bounded_mi = max(0.0, min(100.0, float(raw_mi)))
            return round(bounded_mi, 2)
        except Exception as e:
            logger.debug(f"Radon mi_visit fallback triggered: {e}")
            return cls._calculate_fallback_mi(source_code)

    @classmethod
    def _calculate_fallback_mi(cls, source_code: str) -> float:
        """
        Calculates fallback Maintainability Index from first principles
        (Halstead Volume, McCabe Cyclomatic Complexity, LOC).
        """
        lines = [line.strip() for line in source_code.splitlines() if line.strip() and not line.startswith("#")]
        loc = max(1, len(lines))
        
        # Estimate Cyclomatic Complexity (1 + branching statements)
        cc = 1
        try:
            blocks = cc_visit(source_code)
            if blocks:
                cc = max(1, sum(b.complexity for b in blocks))
        except Exception:
            for line in lines:
                if any(kw in line for kw in ("if ", "for ", "while ", "elif ", "except ")):
                    cc += 1

        # Heuristic Halstead Volume estimation
        tokens = source_code.split()
        n1 = max(1, len(set(tokens)))
        N1 = max(1, len(tokens))
        volume = max(1.0, N1 * math.log2(max(2, n1)))

        # Standard SEI Maintainability Index formula normalized to 0-100
        raw_mi = 171.0 - 5.2 * math.log(volume) - 0.23 * cc - 16.2 * math.log(loc)
        normalized_mi = max(0.0, min(100.0, (raw_mi * 100.0) / 171.0))
        return round(normalized_mi, 2)

    @classmethod
    def generate_labels(
        cls,
        problem_id: str,
        source_code: str,
        custom_time_label: Optional[str] = None,
        custom_space_label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates full ground truth target labels for a given submission.
        
        Args:
            problem_id: Identifier of the problem specification.
            source_code: Cleaned, syntactically valid source code.
            custom_time_label: Optional explicit override for time complexity.
            custom_space_label: Optional explicit override for space complexity.
            
        Returns:
            Dictionary containing:
            - time_complexity_label (str)
            - space_complexity_label (str)
            - maintainability_label (float)
            - code_similarity_group (str)
        """
        spec = get_problem_spec(problem_id)
        
        # 1. Time Complexity Label
        if custom_time_label and custom_time_label in settings.TIME_COMPLEXITY_CLASSES:
            time_label = custom_time_label
        elif spec:
            time_label = spec.time_complexity
        else:
            time_label = "O(N)"  # Standard default assumption if unmapped

        # 2. Space Complexity Label
        if custom_space_label and custom_space_label in settings.SPACE_COMPLEXITY_CLASSES:
            space_label = custom_space_label
        elif spec:
            space_label = spec.space_complexity
        else:
            space_label = "O(1)"

        # 3. Maintainability Label (0.0 to 100.0)
        maintainability = cls.compute_maintainability_index(source_code)

        # 4. Code Similarity Group
        similarity_group = problem_id

        return {
            "time_complexity_label": time_label,
            "space_complexity_label": space_label,
            "maintainability_label": maintainability,
            "code_similarity_group": similarity_group,
        }
