"""
Mathematical utilities module.

This module contains mathematical functions, sampling methods, manifolds, and data augmentation.
"""

from .sampling import sample_gumbel, gumbel_softmax_sample, gumbel_softmax
# from .manifolds import Hyperboloid  # Commented out due to missing dependencies
from .augmentation import (
    continus_mixup_data, mixup_data_by_class, mixup_criterion,
    mixup_cluster_loss, inner_loss, intra_loss
)

__all__ = [
    'sample_gumbel', 'gumbel_softmax_sample', 'gumbel_softmax',
    # 'Hyperboloid',  # Commented out due to missing dependencies
    'continus_mixup_data', 'mixup_data_by_class', 'mixup_criterion',
    'mixup_cluster_loss', 'inner_loss', 'intra_loss'
]
