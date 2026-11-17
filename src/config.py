"""
Project Configuration Module
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Defines project-wide paths, model architectures, feature schemas,
hardware defaults, and runtime safety boundaries.
"""

from pathlib import Path
from typing import List, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project Root Directory Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Environment
    ENV: str = "development"
    SEED: int = 42
    LOG_LEVEL: str = "INFO"

    # Hardware & Performance (Targeting RTX 4050 Laptop GPU - 6GB VRAM)
    DEVICE: str = "auto"  # auto | cuda | cpu
    MIXED_PRECISION: bool = True
    EMBEDDING_BATCH_SIZE: int = 32
    TRAINING_BATCH_SIZE: int = 32
    NUM_WORKERS: int = 0  # 0 recommended on Windows to avoid spawn issues

    # Feature Dimensions
    NUM_AST_FEATURES: int = 25
    CODEBERT_EMBEDDING_DIM: int = 768
    TOTAL_FEATURE_DIM: int = 793  # 25 + 768

    # CodeBERT Specs
    CODEBERT_MODEL_NAME: str = "microsoft/codebert-base"
    MAX_TOKEN_LENGTH: int = 512

    # Storage Paths
    BASE_DIR: Path = PROJECT_ROOT
    DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
    RAW_DATA_DIR: Path = PROJECT_ROOT / "datasets" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "datasets" / "processed"
    EMBEDDINGS_DIR: Path = PROJECT_ROOT / "datasets" / "embeddings"
    MODELS_DIR: Path = PROJECT_ROOT / "saved_models"
    CHECKPOINTS_DIR: Path = PROJECT_ROOT / "saved_models" / "checkpoints"
    BASELINES_DIR: Path = PROJECT_ROOT / "saved_models" / "baselines"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"
    METRICS_DIR: Path = PROJECT_ROOT / "reports" / "metrics"

    # Database
    DATABASE_URL: str = f"sqlite:///{PROJECT_ROOT / 'backend' / 'database' / 'code_evaluation.db'}"

    # Backend API
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    # Security & Safety Limits (Strict Static Analysis - Untrusted Code)
    MAX_CODE_SIZE_BYTES: int = 50000  # ~50 KB limit per submission
    AST_PARSING_TIMEOUT_SECONDS: float = 5.0

    # 25 Structural AST and Metric Features
    AST_FEATURE_NAMES: List[str] = [
        "total_ast_nodes",              # 1. Total count of AST nodes
        "num_functions",                # 2. Function definitions count
        "num_classes",                  # 3. Class definitions count
        "num_variables",                # 4. Assigned variable count
        "num_loops",                    # 5. Total for + while loops
        "num_nested_loops",             # 6. Loops enclosed in other loops
        "max_loop_nesting_depth",       # 7. Maximum depth of loop hierarchy
        "cyclomatic_complexity",        # 8. McCabe cyclomatic complexity (edges - nodes + 2P)
        "num_conditional_statements",   # 9. Total if/elif/else statements
        "num_branching_conditions",     # 10. Total Boolean operators (and/or/not) in branches
        "num_return_statements",        # 11. Count of return statements
        "num_recursive_functions",      # 12. Functions that call themselves
        "num_function_calls",           # 13. Total invocation of functions/methods
        "num_parameters",               # 14. Total formal parameters defined
        "avg_function_length",          # 15. Mean AST statements per function
        "max_function_length",          # 16. Maximum AST statements in any single function
        "num_operators",                # 17. Total mathematical, comparison & bitwise operators
        "num_operands",                 # 18. Total variable and literal identifiers
        "halstead_vocabulary",          # 19. Unique operators + unique operands (n1 + n2)
        "halstead_length",              # 20. Total operators + total operands (N1 + N2)
        "halstead_volume",              # 21. Length * log2(Vocabulary)
        "halstead_difficulty",          # 22. (n1 / 2) * (N2 / n2)
        "halstead_effort",              # 23. Difficulty * Volume
        "variable_reuse_ratio",         # 24. Variable appearances / Unique variables count
        "control_flow_branching_factor" # 25. (Branches + Loops) / Max(1, Functions)
    ]

    # Asymptotic Complexity Classes
    TIME_COMPLEXITY_CLASSES: List[str] = [
        "O(1)",
        "O(log N)",
        "O(N)",
        "O(N log N)",
        "O(N^2)",
        "O(N^3)",
        "O(2^N)"
    ]

    SPACE_COMPLEXITY_CLASSES: List[str] = [
        "O(1)",
        "O(log N)",
        "O(N)",
        "O(N^2)"
    ]


# Instantiate global settings
settings = Settings()
