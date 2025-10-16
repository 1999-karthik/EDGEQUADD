#!/usr/bin/env python3
"""
ALTER Dataset Runner
Run experiments across different ABIDE atlas variants and prepare for ADNI/PPMI support.
"""

import argparse
import sys
import os
import numpy as np
from main import DirectConfig, model_training
from utils.seed import set_seed

def get_available_datasets():
    """Get list of available datasets based on files in datasets directory"""
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

def run_single_experiment(cfg, run_idx, total_runs, base_seed=42):
    """Run a single experiment with given configuration"""
    run_seed = base_seed + run_idx * 1000
    set_seed(run_seed, cfg.get('deterministic', True))
    
    print(f"\nStarting run {run_idx + 1}/{total_runs} with seed {run_seed}...")
    print(f"Dataset: {cfg.dataset.name.upper()}")
    print(f"Model: {cfg.model.name}")
    
    try:
        final_metrics = model_training(cfg)
        if final_metrics:
            print(f"Run {run_idx + 1} completed: AUC={final_metrics['test_auc']:.4f}")
            return final_metrics
        else:
            print(f"Run {run_idx + 1} failed!")
            return None
    except Exception as e:
        print(f"Run {run_idx + 1} failed with error: {e}")
        return None

def run_dataset_experiment(dataset_name, cfg, num_runs=5):
    """Run experiments for a specific dataset"""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENTS FOR DATASET: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Set dataset
    cfg.set_dataset(dataset_name)
    
    # Collect metrics across all runs
    all_test_aucs = []
    all_test_sensitivities = []
    all_test_specificities = []
    all_test_accuracies = []
    
    base_seed = cfg.get('seed', 42)
    
    for run_idx in range(num_runs):
        final_metrics = run_single_experiment(cfg, run_idx, num_runs, base_seed)
        
        if final_metrics:
            all_test_aucs.append(final_metrics['test_auc'])
            all_test_sensitivities.append(final_metrics['test_sensitivity'])
            all_test_specificities.append(final_metrics['test_specificity'])
            all_test_accuracies.append(final_metrics['test_accuracy'])
    
    # Calculate and display results
    if all_test_aucs:
        print(f"\n{'='*80}")
        print(f"RESULTS FOR {dataset_name.upper()}")
        print(f"{'='*80}")
        print(f"Model: {cfg.model.name}")
        print(f"Epochs: {cfg.training.epochs}")
        print(f"Number of runs: {num_runs}")
        print(f"Seeds used: {[base_seed + i * 1000 for i in range(num_runs)]}")
        print("-"*80)
        
        # Calculate statistics
        test_auc_mean = np.mean(all_test_aucs)
        test_auc_std = np.std(all_test_aucs)
        test_sen_mean = np.mean(all_test_sensitivities)
        test_sen_std = np.std(all_test_sensitivities)
        test_spe_mean = np.mean(all_test_specificities)
        test_spe_std = np.std(all_test_specificities)
        test_acc_mean = np.mean(all_test_accuracies)
        test_acc_std = np.std(all_test_accuracies)
        
        print(f"Test ROC-AUC:  {test_auc_mean:.4f} ± {test_auc_std:.4f}")
        print(f"Test Sensitivity: {test_sen_mean:.4f} ± {test_sen_std:.4f}")
        print(f"Test Specificity: {test_spe_mean:.4f} ± {test_spe_std:.4f}")
        print(f"Test Accuracy: {test_acc_mean:.4f} ± {test_acc_std:.4f}")
        print("-"*80)
        print("Individual run results:")
        for i, (auc, sen, spe, acc) in enumerate(zip(all_test_aucs, all_test_sensitivities, all_test_specificities, all_test_accuracies)):
            print(f"  Seed {base_seed + i * 1000}: AUC={auc:.4f}, Sen={sen:.4f}, Spe={spe:.4f}, Acc={acc:.4f}")
        print(f"{'='*80}")
        
        # Save results to CSV file like BioBGT
        os.makedirs("csv", exist_ok=True)
        
        # Create single-line format exactly like BioBGT
        single_line = f"roc:{test_auc_mean * 100:.2f} ± {test_auc_std * 100:.2f},acc:{test_acc_mean:.2f} ± {test_acc_std:.2f},sen:{test_sen_mean * 100:.2f} ± {test_sen_std * 100:.2f},spec:{test_spe_mean * 100:.2f} ± {test_spe_std * 100:.2f},seed:{base_seed},runs:{len(all_test_aucs)},epochs:{cfg.training.epochs},batch_size:{cfg.dataset.batch_size},base_lr:{cfg.optimizer[0].lr},target_lr:{cfg.optimizer[0].lr},wd:{cfg.optimizer[0].weight_decay},layers:{cfg.model.num_layers},activation:relu,dropout:{cfg.model.dropout},pooling:{cfg.model.readout == 'mean'},heads:{cfg.model.num_heads},dim_hidden:{cfg.model.hidden_dim}"
        
        csv_filename = f"csv/{cfg.model.name}_{dataset_name}.csv"
        with open(csv_filename, 'a') as f:
            f.write(single_line + '\n')
        
        print(f"Results saved to: {csv_filename}")
        
        return {
            'dataset': dataset_name,
            'auc_mean': test_auc_mean,
            'auc_std': test_auc_std,
            'sensitivity_mean': test_sen_mean,
            'sensitivity_std': test_sen_std,
            'specificity_mean': test_spe_mean,
            'specificity_std': test_spe_std,
            'accuracy_mean': test_acc_mean,
            'accuracy_std': test_acc_std,
            'num_runs': len(all_test_aucs)
        }
    else:
        print(f"\nNo runs completed successfully for {dataset_name.upper()}!")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run ALTER experiments across multiple datasets")
    
    # Dataset selection
    parser.add_argument("--datasets", nargs='+', default=None,
                       help="Specific datasets to run (e.g., abide_AAL116, adni_schaefer100)")
    parser.add_argument("--list-available", action='store_true',
                       help="List all available datasets and exit")
    
    # Model selection (fixed to GraphTransformer)
    parser.add_argument("--model", default="GraphTransformer", 
                       choices=["GraphTransformer"],
                       help="Model to use (fixed to GraphTransformer)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--repeat", type=int, default=5,
                       help="Number of repeated runs per dataset")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    
    # System parameters
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Model architecture parameters
    parser.add_argument("--hidden_sizes", nargs='+', type=int, default=[360, 100],
                       help="Hidden layer sizes")
    parser.add_argument("--pos_encoding", default="rrwp",
                       choices=["rrwp", "identity", "none"],
                       help="Positional encoding type")
    parser.add_argument("--readout", default="mean",
                       choices=["mean", "sum", "max"],
                       help="Readout function for GraphTransformer")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of model layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--hidden_dim", type=int, default=64,
                       help="Hidden dimension")
    parser.add_argument("--ffn_hidden_dim", type=int, default=128,
                       help="Feed-forward hidden dimension")
    parser.add_argument("--self_attention_layers", type=int, default=2,
                       help="Number of self-attention layers for GraphTransformer")
    
    args = parser.parse_args()
    
    # List available datasets if requested
    if args.list_available:
        available = get_available_datasets()
        print("Available datasets:")
        for dataset in available:
            print(f"  - {dataset}")
        return
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Determine which datasets to run
    if args.datasets:
        datasets_to_run = args.datasets
    else:
        datasets_to_run = get_available_datasets()
    
    # Filter to only available datasets
    available_datasets = get_available_datasets()
    datasets_to_run = [d for d in datasets_to_run if d in available_datasets]
    
    if not datasets_to_run:
        print("No available datasets found!")
        print("Available datasets:", available_datasets)
        return
    
    print(f"Running experiments on {len(datasets_to_run)} datasets: {datasets_to_run}")
    
    # Create base configuration
    cfg = DirectConfig()
    cfg.model.name = args.model
    cfg.training.epochs = args.epochs
    cfg.dataset.batch_size = args.batch_size
    cfg.dataset.test_batch_size = args.batch_size
    cfg.dataset.val_batch_size = args.batch_size
    cfg.optimizer[0].lr = args.lr
    cfg.seed = args.seed
    cfg.model.sizes = args.hidden_sizes
    cfg.model.pos_encoding = args.pos_encoding
    cfg.model.readout = args.readout
    cfg.model.num_layers = args.num_layers
    cfg.model.dropout = args.dropout
    cfg.model.num_heads = args.num_heads
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.ffn_hidden_dim = args.ffn_hidden_dim
    cfg.model.self_attention_layer = args.self_attention_layers
    
    # Run experiments for each dataset
    all_results = []
    for dataset_name in datasets_to_run:
        result = run_dataset_experiment(dataset_name, cfg, args.repeat)
        if result:
            all_results.append(result)
    
    # Print summary of all results
    if all_results:
        print(f"\n{'='*100}")
        print("SUMMARY OF ALL EXPERIMENTS")
        print(f"{'='*100}")
        print(f"{'Dataset':<20} {'AUC (mean±std)':<20} {'Sensitivity':<15} {'Specificity':<15} {'Accuracy':<15} {'Runs':<6}")
        print("-"*100)
        
        for result in all_results:
            print(f"{result['dataset']:<20} "
                  f"{result['auc_mean']:.4f}±{result['auc_std']:.4f}    "
                  f"{result['sensitivity_mean']:.4f}±{result['sensitivity_std']:.4f}  "
                  f"{result['specificity_mean']:.4f}±{result['specificity_std']:.4f}  "
                  f"{result['accuracy_mean']:.4f}±{result['accuracy_std']:.4f}  "
                  f"{result['num_runs']:<6}")
        
        print(f"{'='*100}")
        
        # Find best performing dataset
        best_result = max(all_results, key=lambda x: x['auc_mean'])
        print(f"\nBest performing dataset: {best_result['dataset'].upper()}")
        print(f"Best AUC: {best_result['auc_mean']:.4f} ± {best_result['auc_std']:.4f}")
    else:
        print("\nNo experiments completed successfully!")

if __name__ == "__main__":
    main()
