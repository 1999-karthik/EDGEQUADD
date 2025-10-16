import copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from utils import StandardScaler_crossROI


class StandardScaler:
    """
    Standard scaler for data normalization - similar to ALTER's approach
    """
    def __init__(self, mean: np.array, std: np.array):
        self.mean = mean
        self.std = std

    def transform(self, data: np.array):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.array):
        return (data * self.std) + self.mean


def get_available_datasets(data_dir: str) -> List[str]:
    """
    Get list of available datasets based on files in data directory - like ALTER
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of available dataset names
    """
    import os
    import glob
    
    available = []
    npy_files = glob.glob(f'{data_dir}/*.npy')
    
    for filepath in npy_files:
        filename = os.path.basename(filepath)
        if filename.endswith('.npy'):
            dataset_name = filename[:-4]  # Remove .npy extension
            available.append(dataset_name)
    
    return sorted(available)


def load_data(args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
    """
    Load brain network dataset - unified approach like ALTER
    
    Args:
        args: Arguments object with dataset name
        
    Returns:
        Tuple of (timeseries, correlation_matrix, labels, site_info)
        For datasets without timeseries (like ADHD), timeseries will be None
    """
    # Use ALTER-style dataset naming: dataset_parcellation.npy
    data_path = args.data_dir + '/' + args.dataset + '.npy'
    
    try:
        data = np.load(data_path, allow_pickle=True).item()
    except FileNotFoundError:
        available_datasets = get_available_datasets(args.data_dir)
        raise FileNotFoundError(f"Dataset file not found: {data_path}\n"
                               f"Available datasets: {available_datasets}")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset {args.dataset}: {e}")
    
    # Handle different dataset formats - check for both 'timeseries' and 'timeseires' keys
    if "timeseries" in data:
        # New consolidated datasets with different parcellations
        data_timeseries = data["timeseries"]  # Shape: [subjects, time_points, regions]
        data_pearson = data["correlation"]  # Shape: [subjects, regions, regions]
        data_label = data["labels"]  # 0:control, 1:patient
        site = data.get('site', None)  # May not have site information
        
        # Transpose timeseries from [subjects, time_points, regions] to [subjects, regions, time_points]
        # to match the expected format of the original datasets
        data_timeseries = np.transpose(data_timeseries, (0, 2, 1))  # [subjects, regions, time_points]
    elif "timeseires" in data:
        # Original dataset format
        data_timeseries = data["timeseires"]
        data_pearson = data.get("corr", data.get("adj"))  # Try corr first, then adj
        data_label = data["label"]
        site = data.get('site', None)
    elif "adj" in data or "corr" in data:
        # ADHD dataset format - only correlation matrices, no timeseries
        data_timeseries = None
        data_pearson = data.get("adj", data.get("corr"))  # Try both keys
        data_label = data["label"]
        site = None
        
        # Fix label encoding: ADHD uses 1:control, 0:patient (opposite of ABIDE)
        # Convert to ABIDE format: 0:control, 1:patient
        data_label = 1 - data_label
    else:
        raise KeyError(f"Unknown data format for dataset {args.dataset}. Expected keys: 'timeseries'/'timeseires', 'correlation'/'corr'/'adj', 'labels'/'label'")
    
    # Standardize timeseries data if available
    if data_timeseries is not None:
        data_timeseries = StandardScaler_crossROI(data_timeseries)
    
    # Convert to tensors
    if data_timeseries is not None:
        data_timeseries, data_label, data_pearson = \
            [torch.from_numpy(data).float() for data in (data_timeseries, data_label, data_pearson)]
        return data_timeseries, data_pearson, data_label, site
    else:
        # For datasets without timeseries (like ADHD)
        data_label, data_pearson = \
            [torch.from_numpy(data).float() for data in (data_label, data_pearson)]
        return data_pearson, data_label, site
    

def init_stratified_dataloader(args,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: Optional[np.array] = None,
                               timeseries: Optional[torch.tensor] = None) -> Dict[str, Any]:
    """
    Initialize stratified dataloader - improved version like ALTER
    
    Args:
        args: Arguments object
        final_pearson: Correlation matrices
        labels: Labels tensor
        stratified: Site information for stratification (optional)
        timeseries: Timeseries data (optional)
    
    Returns:
        Dictionary containing train, val, and test dataloaders
    """
    labels = F.one_hot(labels.to(torch.int64))
    length = final_pearson.shape[0]
    train_length = int(length * args.Train_prop)
    val_length = int(length * args.Val_prop)
    test_length = length - train_length - val_length

    # Handle stratification
    if stratified is None:
        # Use labels for stratification
        stratified = labels.argmax(dim=1).cpu().numpy()
    else:
        # Combine site and label for stratification
        y_idx = labels.argmax(dim=1).cpu().numpy()
        stratified = np.array([f"{s}_{y}" for s, y in zip(stratified, y_idx)])

    # First split: train vs (val + test)
    try:
        sss1 = StratifiedShuffleSplit(
            n_splits=1, 
            train_size=train_length, 
            test_size=length-train_length, 
            random_state=args.seed
        )
        for train_index, val_and_test_index in sss1.split(final_pearson, stratified):
            final_pearson_train, labels_train = final_pearson[train_index], labels[train_index]
            final_pearson_val_and_test, labels_val_and_test = final_pearson[val_and_test_index], labels[val_and_test_index]
            stratified_val_test = stratified[val_and_test_index]
            
            # Handle timeseries if available
            if timeseries is not None:
                timeseries_train = timeseries[train_index]
                timeseries_val_and_test = timeseries[val_and_test_index]
            else:
                timeseries_train = None
                timeseries_val_and_test = None
            break
    except ValueError as e:
        print(f"Warning: Stratified split failed, using random split: {e}")
        # Fallback to random split
        train_index, val_and_test_index = train_test_split(
            range(length), 
            test_size=val_length+test_length, 
            train_size=train_length,
            random_state=args.seed
        )
        final_pearson_train, labels_train = final_pearson[train_index], labels[train_index]
        final_pearson_val_and_test, labels_val_and_test = final_pearson[val_and_test_index], labels[val_and_test_index]
        stratified_val_test = None
        
        if timeseries is not None:
            timeseries_train = timeseries[train_index]
            timeseries_val_and_test = timeseries[val_and_test_index]
        else:
            timeseries_train = None
            timeseries_val_and_test = None

    # Second split: val vs test
    if stratified_val_test is not None:
        try:
            sss2 = StratifiedShuffleSplit(
                n_splits=1, 
                test_size=test_length, 
                random_state=args.seed
            )
            for val_index, test_index in sss2.split(final_pearson_val_and_test, stratified_val_test):
                final_pearson_val, labels_val = final_pearson_val_and_test[val_index], labels_val_and_test[val_index]
                final_pearson_test, labels_test = final_pearson_val_and_test[test_index], labels_val_and_test[test_index]
                
                if timeseries_val_and_test is not None:
                    timeseries_val = timeseries_val_and_test[val_index]
                    timeseries_test = timeseries_val_and_test[test_index]
                else:
                    timeseries_val = None
                    timeseries_test = None
                break
        except ValueError as e:
            print(f"Warning: Stratified split failed, using random split: {e}")
            # Fallback to random split
            val_index, test_index = train_test_split(
                range(len(final_pearson_val_and_test)), 
                test_size=test_length, 
                random_state=args.seed
            )
            final_pearson_val, labels_val = final_pearson_val_and_test[val_index], labels_val_and_test[val_index]
            final_pearson_test, labels_test = final_pearson_val_and_test[test_index], labels_val_and_test[test_index]
            
            if timeseries_val_and_test is not None:
                timeseries_val = timeseries_val_and_test[val_index]
                timeseries_test = timeseries_val_and_test[test_index]
            else:
                timeseries_val = None
                timeseries_test = None
    else:
        # Random split for val/test
        val_index, test_index = train_test_split(
            range(len(final_pearson_val_and_test)), 
            test_size=test_length, 
            random_state=args.seed
        )
        final_pearson_val, labels_val = final_pearson_val_and_test[val_index], labels_val_and_test[val_index]
        final_pearson_test, labels_test = final_pearson_val_and_test[test_index], labels_val_and_test[test_index]
        
        if timeseries_val_and_test is not None:
            timeseries_val = timeseries_val_and_test[val_index]
            timeseries_test = timeseries_val_and_test[test_index]
        else:
            timeseries_val = None
            timeseries_test = None

    # Create datasets
    if timeseries_train is not None:
        # With timeseries data
        train_dataset = torch.utils.data.TensorDataset(timeseries_train, final_pearson_train, labels_train)
        val_dataset = torch.utils.data.TensorDataset(timeseries_val, final_pearson_val, labels_val)
        test_dataset = torch.utils.data.TensorDataset(timeseries_test, final_pearson_test, labels_test)
    else:
        # Only correlation matrices
        train_dataset = torch.utils.data.TensorDataset(final_pearson_train, labels_train)
        val_dataset = torch.utils.data.TensorDataset(final_pearson_val, labels_val)
        test_dataset = torch.utils.data.TensorDataset(final_pearson_test, labels_test)

    # Create deterministic generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True, 
        generator=g
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False
    )

    return {
        "train_dataloader": train_dataloader, 
        "val_dataloader": val_dataloader, 
        "test_dataloader": test_dataloader
    }