"""
ALTER Core Utilities

This module contains general utility functions used throughout ALTER.
Organized into logical submodules: data, metrics, and math.
"""

# Import from organized submodules
from .data import (
    load_abide_data, init_dataloader, init_stratified_dataloader,
    StandardScaler, reduce_sample_size, dataset_factory
)
from .metrics import (
    accuracy, isfloat, WeightedMeter, AverageMeter, TotalMeter, count_params
)
from .math import (
    sample_gumbel, gumbel_softmax_sample, gumbel_softmax,
    continus_mixup_data, mixup_data_by_class, mixup_criterion,
    mixup_cluster_loss, inner_loss, intra_loss
)
from .seed import set_seed, get_dataloader_generator, set_worker_seed, DEFAULT_SEED

__all__ = [
    # Data utilities
    'load_abide_data', 'init_dataloader', 'init_stratified_dataloader',
    'StandardScaler', 'reduce_sample_size', 'dataset_factory',
    # Metrics utilities
    'accuracy', 'isfloat', 'WeightedMeter', 'AverageMeter', 'TotalMeter', 'count_params',
    # Math utilities
    'sample_gumbel', 'gumbel_softmax_sample', 'gumbel_softmax',
    'continus_mixup_data', 'mixup_data_by_class', 'mixup_criterion',
    'mixup_cluster_loss', 'inner_loss', 'intra_loss',
    # Seed utilities
    'set_seed', 'get_dataloader_generator', 'set_worker_seed', 'DEFAULT_SEED'
]