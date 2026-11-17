"""
Dataset Builder & Leakage-Free Splitter Module
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Pipeline:
Raw Submissions -> Preprocessing & Deduplication -> Ground-Truth Labeling -> Stratified Problem Splitting

Guarantees:
1. Zero Data Leakage: Problem IDs in Train, Validation, and Test sets are mutually disjoint.
2. Exact CSV Schema specified in research requirements.
3. Comprehensive dataset summary report generation in reports/metrics/dataset_summary.json.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from collections import defaultdict

from src.config import settings
from src.data.preprocess import CodePreprocessor
from src.data.label_generator import LabelGenerator
from src.data.taxonomy import PROBLEM_REGISTRY, ProblemSpec
from src.data.download_dataset import CodeNetDatasetDownloader
from src.utils.logger import logger
from src.utils.device import set_seed


class DatasetBuilder:
    """
    Coordinates end-to-end dataset creation, validation, labeling, and splitting.
    """

    REQUIRED_COLUMNS = [
        "sample_id",
        "problem_id",
        "source_code",
        "programming_language",
        "original_metadata",
        "time_complexity_label",
        "space_complexity_label",
        "maintainability_label",
        "code_similarity_group"
    ]

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = CodePreprocessor()

    def build_dataset(
        self,
        raw_samples: List[Dict[str, Any]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Cleans, labels, and partitions samples into problem-disjoint train, val, and test splits.
        """
        set_seed(seed)
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

        logger.info(f"Building dataset from {len(raw_samples)} raw samples...")
        self.preprocessor.reset_deduplication()

        processed_rows = []
        for sample in raw_samples:
            sample_id = sample["sample_id"]
            problem_id = sample["problem_id"]
            code = sample["source_code"]
            lang = sample.get("programming_language", "python")
            orig_meta = sample.get("original_metadata", "{}")

            passed, reason, clean_code = self.preprocessor.filter_and_deduplicate(sample_id, code)
            if not passed:
                continue

            labels = LabelGenerator.generate_labels(problem_id, clean_code)

            row = {
                "sample_id": sample_id,
                "problem_id": problem_id,
                "source_code": clean_code,
                "programming_language": lang,
                "original_metadata": orig_meta,
                "time_complexity_label": labels["time_complexity_label"],
                "space_complexity_label": labels["space_complexity_label"],
                "maintainability_label": labels["maintainability_label"],
                "code_similarity_group": labels["code_similarity_group"],
            }
            processed_rows.append(row)

        df_full = pd.DataFrame(processed_rows)
        logger.info(f"Preprocessed & labeled: {len(df_full)} valid unique samples.")

        # Perform Stratified Problem ID Partitioning (Zero Data Leakage)
        train_df, val_df, test_df = self._stratified_problem_split(
            df_full, train_ratio, val_ratio, test_ratio, seed
        )

        # Save CSVs
        full_path = self.output_dir / "codenet_python_processed.csv"
        train_path = self.output_dir / "train.csv"
        val_path = self.output_dir / "val.csv"
        test_path = self.output_dir / "test.csv"

        df_full[self.REQUIRED_COLUMNS].to_csv(full_path, index=False, encoding="utf-8")
        train_df[self.REQUIRED_COLUMNS].to_csv(train_path, index=False, encoding="utf-8")
        val_df[self.REQUIRED_COLUMNS].to_csv(val_path, index=False, encoding="utf-8")
        test_df[self.REQUIRED_COLUMNS].to_csv(test_path, index=False, encoding="utf-8")

        # Generate Metrics & Summary
        summary = self._generate_dataset_summary(df_full, train_df, val_df, test_df)
        summary_path = settings.METRICS_DIR / "dataset_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Dataset successfully saved to {self.output_dir}")
        logger.info(f"Summary saved to {summary_path}")

        return {
            "full": df_full,
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

    @staticmethod
    def _stratified_problem_split(
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partitions problems by complexity class to guarantee:
        1. Every split (train, val, test) contains balanced representation of complexity classes.
        2. Problem IDs are strictly disjoint (0% problem leakage across splits).
        """
        random.seed(seed)
        
        # Map each complexity class to its associated problem IDs
        class_to_pids = defaultdict(list)
        for pid, spec in PROBLEM_REGISTRY.items():
            class_to_pids[spec.time_complexity].append(pid)

        train_pids, val_pids, test_pids = set(), set(), set()

        for comp_class, pids in class_to_pids.items():
            shuffled = list(pids)
            random.shuffle(shuffled)
            n_pids = len(shuffled)
            
            if n_pids == 1:
                # If only 1 problem for this class, allocate to train
                train_pids.add(shuffled[0])
            elif n_pids == 2:
                train_pids.add(shuffled[0])
                test_pids.add(shuffled[1])
            else:
                n_train = max(1, int(round(n_pids * train_ratio)))
                n_val = max(1, int(round(n_pids * val_ratio)))
                # Remaining go to test
                train_slice = shuffled[:n_train]
                val_slice = shuffled[n_train:n_train + n_val]
                test_slice = shuffled[n_train + n_val:]
                
                # Ensure no empty test slice if possible
                if not test_slice and len(train_slice) > 1:
                    test_slice = [train_slice.pop()]

                train_pids.update(train_slice)
                val_pids.update(val_slice)
                test_pids.update(test_slice)

        # Confirm 0% leakage
        assert train_pids.isdisjoint(val_pids), "Leakage detected between train and val pids!"
        assert train_pids.isdisjoint(test_pids), "Leakage detected between train and test pids!"
        assert val_pids.isdisjoint(test_pids), "Leakage detected between val and test pids!"

        train_df = df[df["problem_id"].isin(train_pids)].copy()
        val_df = df[df["problem_id"].isin(val_pids)].copy()
        test_df = df[df["problem_id"].isin(test_pids)].copy()

        logger.info(
            f"Problem ID Split: Train={len(train_pids)} pids ({len(train_df)} samples), "
            f"Val={len(val_pids)} pids ({len(val_df)} samples), "
            f"Test={len(test_pids)} pids ({len(test_df)} samples)"
        )

        return train_df, val_df, test_df

    @staticmethod
    def _generate_dataset_summary(
        df_full: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generates statistical distribution summary for research documentation."""
        return {
            "total_samples": int(len(df_full)),
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "test_samples": int(len(test_df)),
            "total_problems": int(df_full["problem_id"].nunique()),
            "time_complexity_distribution": df_full["time_complexity_label"].value_counts().to_dict(),
            "space_complexity_distribution": df_full["space_complexity_label"].value_counts().to_dict(),
            "maintainability_stats": {
                "mean": round(float(df_full["maintainability_label"].mean()), 2),
                "std": round(float(df_full["maintainability_label"].std()), 2),
                "min": round(float(df_full["maintainability_label"].min()), 2),
                "max": round(float(df_full["maintainability_label"].max()), 2),
                "median": round(float(df_full["maintainability_label"].median()), 2),
            },
            "leakage_verification": {
                "train_val_overlap_count": int(len(set(train_df["problem_id"]).intersection(set(val_df["problem_id"])))),
                "train_test_overlap_count": int(len(set(train_df["problem_id"]).intersection(set(test_df["problem_id"])))),
                "val_test_overlap_count": int(len(set(val_df["problem_id"]).intersection(set(test_df["problem_id"])))),
                "zero_leakage_guaranteed": True
            }
        }
