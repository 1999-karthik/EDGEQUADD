"""
Data utilities module.

This module contains dataset loading, preprocessing, and data loading utilities.
"""

from .datasets import (load_abide_data, load_adni_data, load_adhd_data, load_ppmi_data, 
                       load_taowu_data, load_neurocon_data, load_aal116_data, load_harvard48_data, 
                       load_kmeans100_data, load_schaefer100_data, load_ward100_data)
from .dataloader import init_dataloader, init_stratified_dataloader
from .preprocessing import StandardScaler, reduce_sample_size

# Dataset factory function
from omegaconf import DictConfig
from typing import List
import torch.utils as utils

def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:
    """Factory function to create datasets based on configuration."""
    # Choose the appropriate loading function based on dataset name
    dataset_name = cfg.dataset.name.lower()
    
    if dataset_name.startswith('ppmi'):
        datasets = load_ppmi_data(cfg)
    elif dataset_name.startswith('adni'):
        datasets = load_adni_data(cfg)
    elif dataset_name.startswith('adhd'):
        datasets = load_adhd_data(cfg)
    elif dataset_name.startswith('taowu'):
        datasets = load_taowu_data(cfg)
    elif dataset_name.startswith('neurocon'):
        datasets = load_neurocon_data(cfg)
    elif dataset_name.startswith('aal116'):
        datasets = load_aal116_data(cfg)
    elif dataset_name.startswith('harvard48'):
        datasets = load_harvard48_data(cfg)
    elif dataset_name.startswith('kmeans100'):
        datasets = load_kmeans100_data(cfg)
    elif dataset_name.startswith('schaefer100'):
        datasets = load_schaefer100_data(cfg)
    elif dataset_name.startswith('ward100'):
        datasets = load_ward100_data(cfg)
    else:
        # Default to ABIDE for all other datasets
        datasets = load_abide_data(cfg)
    
    dataloaders = init_stratified_dataloader(cfg, *datasets) \
        if cfg.dataset.stratified \
        else init_dataloader(cfg, *datasets)
    
    return dataloaders

__all__ = [
    'load_abide_data', 'load_adni_data', 'load_adhd_data', 'load_ppmi_data',
    'load_taowu_data', 'load_neurocon_data', 'load_aal116_data', 'load_harvard48_data',
    'load_kmeans100_data', 'load_schaefer100_data', 'load_ward100_data', 'load_ppmi_data',
    'load_aal116_data', 'load_harvard48_data', 'load_kmeans100_data', 
    'load_schaefer100_data', 'load_ward100_data',
    'init_dataloader', 
    'init_stratified_dataloader',
    'StandardScaler', 
    'reduce_sample_size',
    'dataset_factory'
]
