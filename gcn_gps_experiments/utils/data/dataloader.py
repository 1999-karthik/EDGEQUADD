import torch
import torch.utils.data as utils
from omegaconf import DictConfig

# Custom open_dict function for DirectConfig compatibility
def open_dict(cfg):
    """Compatibility function for open_dict() with DirectConfig"""
    return cfg
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F
from ..seed import get_dataloader_generator, set_worker_seed


def init_dataloader(cfg: DictConfig,
                    final_timeseires: torch.tensor,
                    final_pearson: torch.tensor,
                    labels: torch.tensor) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    dataset = utils.TensorDataset(
        final_timeseires[:train_length+val_length+test_length],
        final_pearson[:train_length+val_length+test_length],
        labels[:train_length+val_length+test_length]
    )

    # Create deterministic generator for reproducible splits
    generator = get_dataloader_generator(cfg.get('seed', 42))
    
    train_dataset, val_dataset, test_dataset = utils.random_split(
        dataset, [train_length, val_length, test_length], generator=generator)
    
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, 
        drop_last=cfg.dataset.drop_last, generator=generator,
        worker_init_fn=set_worker_seed)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, 
        drop_last=False, generator=generator,
        worker_init_fn=set_worker_seed)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, 
        drop_last=False, generator=generator,
        worker_init_fn=set_worker_seed)

    return [train_dataloader, val_dataloader, test_dataloader]


def init_stratified_dataloader(cfg: DictConfig,
                               final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: np.array) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        cfg.steps_per_epoch = (train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    # Handle case when stratified (site) is None - use random split instead
    if stratified is None:
        from sklearn.model_selection import train_test_split
        train_index, test_valid_index = train_test_split(
            range(length), test_size=val_length+test_length, train_size=train_length,
            random_state=cfg.get('seed', 42))
        final_timeseires_train, final_pearson_train, labels_train = final_timeseires[
            train_index], final_pearson[train_index], labels[train_index]
        final_timeseires_val_test, final_pearson_val_test, labels_val_test = final_timeseires[
            test_valid_index], final_pearson[test_valid_index], labels[test_valid_index]
    else:
        split = StratifiedShuffleSplit(
            n_splits=1, test_size=val_length+test_length, train_size=train_length, 
            random_state=cfg.get('seed', 42))
        for train_index, test_valid_index in split.split(final_timeseires, stratified):
            final_timeseires_train, final_pearson_train, labels_train = final_timeseires[
                train_index], final_pearson[train_index], labels[train_index]
            final_timeseires_val_test, final_pearson_val_test, labels_val_test = final_timeseires[
                test_valid_index], final_pearson[test_valid_index], labels[test_valid_index]
            stratified = stratified[test_valid_index]

    # Handle second split for validation/test
    if stratified is None:
        test_index, valid_index = train_test_split(
            range(len(final_timeseires_val_test)), test_size=test_length, 
            random_state=cfg.get('seed', 42))
        final_timeseires_test, final_pearson_test, labels_test = final_timeseires_val_test[
            test_index], final_pearson_val_test[test_index], labels_val_test[test_index]
        final_timeseires_val, final_pearson_val, labels_val = final_timeseires_val_test[
            valid_index], final_pearson_val_test[valid_index], labels_val_test[valid_index]
    else:
        split2 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_length, random_state=cfg.get('seed', 42))
        for test_index, valid_index in split2.split(final_timeseires_val_test, stratified):
            final_timeseires_test, final_pearson_test, labels_test = final_timeseires_val_test[
                test_index], final_pearson_val_test[test_index], labels_val_test[test_index]
            final_timeseires_val, final_pearson_val, labels_val = final_timeseires_val_test[
                valid_index], final_pearson_val_test[valid_index], labels_val_test[valid_index]

    train_dataset = utils.TensorDataset(
        final_timeseires_train,
        final_pearson_train,
        labels_train
    )

    val_dataset = utils.TensorDataset(
        final_timeseires_val, final_pearson_val, labels_val
    )

    test_dataset = utils.TensorDataset(
        final_timeseires_test, final_pearson_test, labels_test
    )

    # Create deterministic generator for reproducible data loading
    generator = get_dataloader_generator(cfg.get('seed', 42))
    
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, 
        drop_last=cfg.dataset.drop_last, generator=generator,
        worker_init_fn=set_worker_seed)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, 
        drop_last=False, generator=generator,
        worker_init_fn=set_worker_seed)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, 
        drop_last=False, generator=generator,
        worker_init_fn=set_worker_seed)

    return [train_dataloader, val_dataloader, test_dataloader]
