import numpy as np
import torch
from .preprocessing import StandardScaler
from omegaconf import DictConfig

# Custom open_dict function for DirectConfig compatibility
def open_dict(cfg):
    """Compatibility function for open_dict() with DirectConfig"""
    return cfg


def load_abide_data(cfg: DictConfig):

    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    
    # Handle different dataset formats
    # Check if this is a consolidated dataset (has 'timeseries' key) or original format (has 'timeseires' key)
    if "timeseries" in data:
        # New consolidated datasets with different parcellations
        final_timeseires = data["timeseries"]  # Shape: [subjects, time_points, regions]
        final_pearson = data["correlation"]  # Shape: [subjects, regions, regions]
        labels = data["labels"]  # 0:control, 1:patient
        site = None  # New datasets don't have site information
        
        # Transpose timeseries from [subjects, time_points, regions] to [subjects, regions, time_points]
        # to match the expected format of the original ABIDE dataset
        final_timeseires = np.transpose(final_timeseires, (0, 2, 1))  # [subjects, regions, time_points]
    else:
        # Original ABIDE dataset format
        final_timeseires = data["timeseires"]
        final_pearson = data["corr"]
        labels = data["label"]
        site = data['site']

    scaler = StandardScaler(mean=np.mean(
        final_timeseires), std=np.std(final_timeseires))

    final_timeseires = scaler.transform(final_timeseires)

    final_timeseires, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels)]

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]

    return final_timeseires, final_pearson, labels, site


def load_adni_data(cfg: DictConfig):
    """Load ADNI dataset - same format as ABIDE"""
    return load_abide_data(cfg)


def load_adhd_data(cfg: DictConfig):
    """Load ADHD dataset - same format as ABIDE"""
    return load_abide_data(cfg)


def load_ppmi_data(cfg: DictConfig):
    """Load PPMI dataset with ALL classes (multi-class classification)"""
    # Load PPMI data directly (not through ABIDE format)
    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    
    # PPMI format: timeseries, correlation, labels
    if "timeseries" not in data:
        raise KeyError("PPMI dataset missing 'timeseries' key. Expected PPMI format.")
    if "correlation" not in data:
        raise KeyError("PPMI dataset missing 'correlation' key. Expected PPMI format.")
    if "labels" not in data:
        raise KeyError("PPMI dataset missing 'labels' key. Expected PPMI format.")
    
    # Extract data - NO FILTERING, keep all classes
    timeseries_data = data["timeseries"]  # Shape: [subjects, time_points, regions]
    correlation_data = data["correlation"]  # Shape: [subjects, regions, regions]
    labels_data = data["labels"]  # [subjects,], 0:control, 1:PD, 2:SWEDD, 3:Prodromal
    site = data.get('site', None)
    
    print(f"PPMI Data Loading (ALL CLASSES):")
    print(f"  Total samples: {len(labels_data)}")
    print(f"  Control samples (0): {np.sum(labels_data == 0)}")
    print(f"  PD samples (1): {np.sum(labels_data == 1)}")
    print(f"  SWEDD samples (2): {np.sum(labels_data == 2)}")
    print(f"  Prodromal samples (3): {np.sum(labels_data == 3)}")
    print(f"  Unique labels: {np.unique(labels_data)}")
    
    # Transpose timeseries from [subjects, time_points, regions] to [subjects, regions, time_points]
    # to match the expected format of the original ABIDE dataset
    final_timeseires = np.transpose(timeseries_data, (0, 2, 1))  # [subjects, regions, time_points]
    final_pearson = correlation_data
    
    # Apply standardization
    scaler = StandardScaler(mean=np.mean(final_timeseires), std=np.std(final_timeseires))
    final_timeseires = scaler.transform(final_timeseires)
    
    # Convert to tensors
    final_timeseires, final_pearson, labels = [torch.from_numpy(data).float() for data in (final_timeseires, final_pearson, labels_data)]
    
    # Update config
    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]
    
    return final_timeseires, final_pearson, labels, site


def load_taowu_data(cfg: DictConfig):
    """Load TAOWU dataset - same format as ABIDE"""
    return load_abide_data(cfg)


def load_neurocon_data(cfg: DictConfig):
    """Load NEUROCON dataset - same format as ABIDE"""
    return load_abide_data(cfg)


def load_abcd_data(cfg: DictConfig):
    """Load ABCD dataset - same format as ABIDE"""
    return load_abide_data(cfg)


def load_aal116_data(cfg: DictConfig):
    """Load AAL116 dataset - same format as ABIDE"""
    return load_abide_data(cfg)


def load_harvard48_data(cfg: DictConfig):
    """Load HARVARD48 dataset - same format as ABIDE"""
    return load_abide_data(cfg)


def load_kmeans100_data(cfg: DictConfig):
    """Load KMEANS100 dataset - same format as ABIDE"""
    return load_abide_data(cfg)


def load_schaefer100_data(cfg: DictConfig):
    """Load SCHAEFER100 dataset - same format as ABIDE"""
    return load_abide_data(cfg)


def load_ward100_data(cfg: DictConfig):
    """Load WARD100 dataset - same format as ABIDE"""
    return load_abide_data(cfg)
