"""
Hardware & Device Management Module
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Optimized for NVIDIA RTX 4050 Laptop GPU (6 GB VRAM) & Intel Core i7-13700HX.
Provides deterministic seeding, memory diagnostics, and mixed-precision helpers.
"""

import os
import random
from typing import Dict, Any, Optional
import numpy as np
import torch

from src.utils.logger import logger


def set_seed(seed: int = 42) -> None:
    """
    Sets global deterministic seeds across standard library random,
    NumPy, and PyTorch (CPU and all CUDA devices).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Guarantees reproducible convolutions at minor runtime overhead
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug(f"Deterministic seed set to {seed}")


def get_device(preferred: str = "auto") -> torch.device:
    """
    Resolves execution device based on hardware availability and user preference.
    
    Args:
        preferred: 'auto', 'cuda', or 'cpu'.
        
    Returns:
        torch.device instance.
    """
    if preferred == "cpu":
        device = torch.device("cpu")
    elif preferred in ("cuda", "auto"):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            if preferred == "cuda":
                logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device preference: {preferred}")

    return device


def get_hardware_profile() -> Dict[str, Any]:
    """
    Inspects and returns a diagnostic dictionary of available compute resources.
    """
    cuda_available = torch.cuda.is_available()
    profile: Dict[str, Any] = {
        "cuda_available": cuda_available,
        "pytorch_version": torch.__version__,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
    }

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        profile.update({
            "gpu_name": gpu_name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "total_vram_gb": round(total_vram_gb, 2),
            "supports_fp16": capability[0] >= 7,  # Volta and above support Tensor Cores
            "supports_bf16": torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else False,
        })
    else:
        profile.update({
            "gpu_name": "N/A (CPU Only)",
            "compute_capability": "N/A",
            "total_vram_gb": 0.0,
            "supports_fp16": False,
            "supports_bf16": False,
        })

    return profile


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Returns current GPU memory allocation and reservation in Megabytes (MB).
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
    }


def clear_gpu_cache() -> None:
    """Releases unallocated cached memory from PyTorch CUDA memory allocator."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
