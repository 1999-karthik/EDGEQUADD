"""
ALTER Training Module

This module contains training logic and components like optimizers, schedulers, and callbacks.
"""

from .optimizer import optimizers_factory
from .lr_scheduler import lr_scheduler_factory, LRScheduler
from .logger import logger_factory
from .training import Train

# Training factory function
from omegaconf import DictConfig
from typing import List
import torch
import torch.utils.data as utils
import logging

def training_factory(config: DictConfig,
                     model: torch.nn.Module,
                     optimizers: List[torch.optim.Optimizer],
                     lr_schedulers: List[LRScheduler],
                     dataloaders: List[utils.DataLoader],
                     logger) -> Train:
    """Factory function to create training objects based on configuration."""
    train = config.model.get("train", None)
    if not train:
        train = config.training.name
    return eval(train)(cfg=config,
                       model=model,
                       optimizers=optimizers,
                       lr_schedulers=lr_schedulers,
                       dataloaders=dataloaders,
                       logger=logger)

__all__ = [
    'optimizers_factory',
    'lr_scheduler_factory', 'LRScheduler',
    'logger_factory',
    'Train',
    'training_factory'
]