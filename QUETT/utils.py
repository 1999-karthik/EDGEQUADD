import numpy as np
import torch
import random
import torch.nn as nn
import math
from typing import List
import logging

def continues_mixup_data(*xs, y=None, alpha=1.0, device='cuda'):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    y = lam * y + (1-lam) * y[index]
    return *new_xs, y


def accuracy(output: torch.Tensor, target: torch.Tensor, top_k=(1,)) -> List[float]:
    """Computes the precision@k for the specified values of k ; which is in BNT"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, predict = output.topk(max_k, 1, True, True)
    predict = predict.t()
    correct = predict.eq(target.view(1, -1).expand_as(predict))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def StandardScaler_crossROI(data):
    """
    Standardize data across ROIs (regions of interest).
    
    Args:
        data: numpy array of shape (N, T, R) where N is number of subjects,
              T is time points, R is number of ROIs
    
    Returns:
        Standardized data with same shape
    """
    # Reshape data to (N*R, T) for standardization across time
    N, T, R = data.shape
    data_reshaped = data.transpose(0, 2, 1).reshape(-1, T)  # (N*R, T)
    
    # Standardize across time dimension
    mean = np.mean(data_reshaped, axis=1, keepdims=True)
    std = np.std(data_reshaped, axis=1, keepdims=True)
    std[std == 0] = 1  # Avoid division by zero
    
    data_standardized = (data_reshaped - mean) / std
    
    # Reshape back to original shape (N, T, R)
    data_standardized = data_standardized.reshape(N, R, T).transpose(0, 2, 1)
    
    return data_standardized


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Make cuDNN deterministic
    torch.backends.cudnn.benchmark = False
    # Additional seed fixes for more reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Fix CuBLAS warnings
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def hyper_para_load(args, dataset):
    """
    load hyper parameters of model
    """
    # Extract parameters from dataset automatically
    # For ADHD: dataset = (data_pearson, data_label, site)
    # For other datasets: dataset = (data_timeseries, data_pearson, data_label, site)
    if len(dataset) == 3:  # ADHD case - no timeseries
        correlation_matrices = dataset[0]  # data_pearson at index 0
        labels = dataset[1]  # data_label at index 1
    else:  # Other datasets - with timeseries
        correlation_matrices = dataset[1]  # data_pearson at index 1
        labels = dataset[2]  # data_label at index 2
    
    node_sz = correlation_matrices.shape[1]  # ROI number of each subject
    node_feature_sz = correlation_matrices.shape[-1]  # dim of correlation matrix
    
    # Detect number of classes from labels
    if labels.dim() > 1:  # One-hot encoded
        num_classes = labels.shape[1]
    else:  # Integer labels
        num_classes = len(torch.unique(labels))
    
    print(f"Auto-detected parameters:")
    print(f"  Node size (ROI number): {node_sz}")
    print(f"  Node feature size: {node_feature_sz}")
    print(f"  Number of classes: {num_classes}")

    layers = args.layers
    dropout = args.dropout

    pooling = args.pooling
    cluster_num = args.cluster_num

    orthogonal = True
    freeze_center = True
    project_assignment = True

    return (node_sz, node_feature_sz, layers, dropout,
            pooling, cluster_num, num_classes)


def count_param(model: nn.Module):
    total_parameters = sum(p.numel() for p in model.parameters())
    return total_parameters


def optimizer_update(optimizer: torch.optim.Optimizer, step: int, total_steps: int, args):
    base_lr = args.base_lr
    target_lr = base_lr  # Use same as base_lr for cosine annealing
    total_steps = total_steps

    current_ratio = step / total_steps
    cosine = math.cos(math.pi * current_ratio)
    lr = target_lr + (base_lr - target_lr) * (1 + cosine) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def initialize_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger