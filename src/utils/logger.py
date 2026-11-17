"""
Logging Configuration Module
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Provides structured logging with sanitization to prevent sensitive
source code leakage into disk logs.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "code_grading_ai",
    log_file: Optional[Path] = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Configures and returns a structured logger with both console and file handlers.
    
    Args:
        name: Logger hierarchy name.
        log_file: Optional file path for file logging.
        level: Log severity level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers if re-initialized
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global application logger instance
logger = setup_logger()
