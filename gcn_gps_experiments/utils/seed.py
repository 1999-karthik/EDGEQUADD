"""
Seed management utilities for reproducibility.

This module provides functions to set random seeds for all major libraries
to ensure reproducible results across runs.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set deterministic algorithms
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_dataloader_generator(seed: int) -> torch.Generator:
    """
    Create a deterministic generator for DataLoader.
    
    Args:
        seed: Random seed value
        
    Returns:
        PyTorch generator for deterministic data loading
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def set_worker_seed(worker_id: int) -> None:
    """
    Set seed for DataLoader workers.
    
    Args:
        worker_id: Worker ID for multi-process data loading
    """
    # Use a different seed for each worker but keep it deterministic
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Default seed for the project
DEFAULT_SEED = 42
