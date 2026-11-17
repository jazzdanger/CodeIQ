"""
Phase 2 Dataset Generation CLI
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Executes the complete Phase 2 dataset generation pipeline:
1. Generates/Ingests 5,000 Python competitive programming submissions.
2. Applies strict non-executing static AST validation & deduplication.
3. Computes ground-truth Big-O Time & Space complexity and Radon Maintainability Index.
4. Partitions into Problem-Disjoint Train (70%), Validation (15%), and Test (15%) splits.
5. Saves CSVs and generates research dataset summary report.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_dataset import CodeNetDatasetDownloader
from src.data.dataset_builder import DatasetBuilder
from src.utils.logger import logger


def run_pipeline(target_samples: int = 5000):
    logger.info("=" * 60)
    logger.info(f"STARTING PHASE 2 DATASET GENERATION (Target: {target_samples} samples)")
    logger.info("=" * 60)

    # 1. Download / Corpus Ingestion
    downloader = CodeNetDatasetDownloader()
    raw_corpus = downloader.generate_algorithmic_corpus(target_samples=target_samples)

    # 2. Build Dataset & Stratified Problem Split
    builder = DatasetBuilder()
    splits = builder.build_dataset(raw_corpus, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    df_full = splits["full"]
    df_train = splits["train"]
    df_val = splits["val"]
    df_test = splits["test"]

    logger.info("=" * 60)
    logger.info("PHASE 2 DATASET GENERATION SUMMARY:")
    logger.info(f"- Full Clean Submissions : {len(df_full)}")
    logger.info(f"- Training Set (70%)     : {len(df_train)} submissions ({df_train['problem_id'].nunique()} problems)")
    logger.info(f"- Validation Set (15%)   : {len(df_val)} submissions ({df_val['problem_id'].nunique()} problems)")
    logger.info(f"- Test Set (15%)         : {len(df_test)} submissions ({df_test['problem_id'].nunique()} problems)")
    logger.info("=" * 60)
    logger.info("Time Complexity Distribution:")
    for cls_name, count in df_full["time_complexity_label"].value_counts().items():
        logger.info(f"  {cls_name:10s} : {count} ({count/len(df_full)*100:.1f}%)")
    logger.info("=" * 60)
    logger.info("Space Complexity Distribution:")
    for cls_name, count in df_full["space_complexity_label"].value_counts().items():
        logger.info(f"  {cls_name:10s} : {count} ({count/len(df_full)*100:.1f}%)")
    logger.info("=" * 60)
    logger.info("Maintainability Index Statistics (Radon MI 0-100):")
    logger.info(f"  Mean   : {df_full['maintainability_label'].mean():.2f}")
    logger.info(f"  Std    : {df_full['maintainability_label'].std():.2f}")
    logger.info(f"  Min/Max: {df_full['maintainability_label'].min():.2f} / {df_full['maintainability_label'].max():.2f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline(target_samples=5000)
