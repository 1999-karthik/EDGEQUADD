from datetime import datetime
from utils import dataset_factory
from models import model_factory
from training import lr_scheduler_factory, optimizers_factory
from training import training_factory
from utils.seed import set_seed

# Custom open_dict function for DirectConfig compatibility
def open_dict(cfg):
    """Compatibility function for open_dict() with DirectConfig"""
    return cfg

class DirectConfig:
    """Direct parameter configuration class with YAML defaults"""
    def __init__(self):
        # Dataset parameters (default: ABIDE)
        # Available datasets: 'abide', 'adni', 'adhd', 'ppmi'
        class DatasetConfig:
            def __init__(self):
                self.name = 'abide'  # Change this to switch datasets
                self.batch_size = 16
                self.test_batch_size = 16
                self.val_batch_size = 16
                self.train_set = 0.7
                self.val_set = 0.1
                self.path = 'datasets/abide.npy'  # Will be updated based on dataset name
                self.stratified = True
                self.drop_last = True
                self.node_sz = 116  # Default for AAL116
                self.node_feature_sz = 116  # Default for AAL116
                self.timeseries_sz = 200  # Default timeseries length
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        self.dataset = DatasetConfig()
        
        # Model parameters for GCN and GPS
        class ModelConfig:
            def __init__(self):
                self.name = 'GCN'  # Options: 'GCN', 'GPS'
                
                # GCN specific parameters
                self.hidden_dim = 64
                self.num_layers = 3
                self.dropout = 0.5
                self.pooling = 'mean'  # mean, max, add
                self.threshold = 0.3  # Same as ALTER for adjacency matrix conversion
                
                # GPS specific parameters (same as ALTER)
                self.pe_dim = 32
                self.walk_length = 8  # RRWP walk length
                self.use_positional_encoding = True
                self.threshold = 0.3  # Same as ALTER for adjacency matrix conversion
                
                # General model parameters
                self.activation = 'relu'
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        self.model = ModelConfig()
        
        # Training parameters (from basic_training.yaml)
        class TrainingConfig:
            def __init__(self):
                self.name = 'Train'
                self.epochs = 200
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        self.training = TrainingConfig()
        
        # Optimizer parameters (from adam.yaml) - should be a list
        class LRSchedulerConfig:
            def __init__(self):
                self.mode = 'cos'
                self.base_lr = 1.0e-4
                self.target_lr = 1.0e-5
                self.decay_factor = 0.1
                self.milestones = [0.3, 0.6, 0.9]
                self.poly_power = 2.0
                self.lr_decay = 0.98
                self.warm_up_from = 0.0
                self.warm_up_steps = 0
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        class OptimizerConfig:
            def __init__(self):
                self.name = 'Adam'
                self.lr = 1.0e-4
                self.match_rule = None
                self.except_rule = None
                self.no_weight_decay = False
                self.weight_decay = 1.0e-4
                self.lr_scheduler = LRSchedulerConfig()
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        self.optimizer = [OptimizerConfig()]
        
        # Data size parameters (from 100p.yaml)
        class DataSizeConfig:
            def __init__(self):
                self.percentage = 1.0
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        self.datasz = DataSizeConfig()
        
        # Preprocessing parameters (from mixup.yaml)
        class PreprocessConfig:
            def __init__(self):
                self.name = 'continus_mixup'
                self.continus = True
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        self.preprocess = PreprocessConfig()
        
        # Other parameters
        self.repeat_time = 5  # Fixed at 5 seeds per protocol
        self.seed = 42
        self.deterministic = True
        self.project = 'graphtransformer'
        self.unique_id = None
        
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def set_dataset(self, dataset_name):
        """Switch to a different dataset - supports dataset_parcellation format"""
        self.dataset.name = dataset_name
        
        # Use direct .npy file paths for all datasets
        self.dataset.path = f'datasets/{dataset_name}.npy'
        
        print(f"Switched to dataset: {dataset_name.upper()}")
    
    def get_available_datasets(self):
        """Get list of available datasets based on files in datasets directory"""
        import os
        import glob
        datasets_dir = 'datasets'
        available = []
        
        # Check for all .npy files in datasets directory
        npy_files = glob.glob(f'{datasets_dir}/*.npy')
        
        for filepath in npy_files:
            filename = os.path.basename(filepath)
            if filename.endswith('.npy'):
                dataset_name = filename[:-4]  # Remove .npy extension
                available.append(dataset_name)
        
        return sorted(available)
    
    def _get_node_flag(self, flag):
        """Compatibility method for open_dict()"""
        return False
    
    def _set_flag(self, flag, value):
        """Compatibility method for open_dict()"""
        pass
    
    def __enter__(self):
        """Context manager entry for open_dict() compatibility"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit for open_dict() compatibility"""
        pass


def model_training(cfg):
    """Train the model and return final metrics"""
    try:
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

        # Initialize components
        dataloaders = dataset_factory(cfg)
        model = model_factory(cfg)
        optimizers = optimizers_factory(
            model=model, optimizer_configs=cfg.optimizer)
        lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                             cfg=cfg)
        training = training_factory(cfg, model, optimizers,
                                    lr_schedulers, dataloaders, None)

        # Train and get final metrics
        final_metrics = training.train()
        return final_metrics
        
    except Exception as e:
        import traceback
        print(f"Training failed: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None


# Main.py contains only the core configuration and training functions
# Use parser.py to run experiments with command line arguments
