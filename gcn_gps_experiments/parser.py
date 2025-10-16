#!/usr/bin/env python3
"""
GraphTransformer - Brain Network Analysis
Run GraphTransformer experiments with dynamic parameters.
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from main import DirectConfig, model_training
from utils.seed import set_seed

def save_results_to_csv(dataset_name, model_name, all_test_aucs, all_test_sensitivities, 
                       all_test_specificities, all_test_accuracies, base_seed, repeat_time):
    """
    Save experiment results to CSV file in BrainNetCNN format.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        all_test_aucs: List of test AUC values
        all_test_sensitivities: List of test sensitivity values
        all_test_specificities: List of test specificity values
        all_test_accuracies: List of test accuracy values
        base_seed: Base seed used for experiments
        repeat_time: Number of repeated runs
    """
    # Calculate mean and std for each metric
    auc_mean = np.mean(all_test_aucs)
    auc_std = np.std(all_test_aucs)
    sen_mean = np.mean(all_test_sensitivities)
    sen_std = np.std(all_test_sensitivities)
    spe_mean = np.mean(all_test_specificities)
    spe_std = np.std(all_test_specificities)
    acc_mean = np.mean(all_test_accuracies)
    acc_std = np.std(all_test_accuracies)
    
    # Create single-line format matching BrainNetCNN
    single_line = f"macro_auc:{auc_mean * 100:.2f} ± {auc_std * 100:.2f},macro_recall:{sen_mean * 100:.2f} ± {sen_std * 100:.2f},macro_specificity:{spe_mean * 100:.2f} ± {spe_std * 100:.2f},accuracy:{acc_mean:.2f} ± {acc_std:.2f},seed:{base_seed},runs:{repeat_time},epochs:2,batch_size:16,base_lr:0.0001,target_lr:0.0001,wd:0.0001,layers:3,activation:relu,dropout:0.5,pooling:True,heads:4,dim_hidden:64,model:{model_name}"
    
    # Add macro_metrics flag for PPMI datasets
    if dataset_name.lower().startswith('ppmi'):
        single_line += ",macro_metrics:True"
    
    # Save to CSV in single-line format
    os.makedirs("csv", exist_ok=True)
    csv_filename = f"csv/{model_name}_{dataset_name}.csv"
    with open(csv_filename, 'a') as f:
        f.write(single_line + '\n')
    
    return csv_filename

def main():
    parser = argparse.ArgumentParser(description="Run GraphTransformer experiments")
    
    # Dataset selection
    parser.add_argument("--dataset", default="ppmi_AAL116",
                       help="Dataset to use (e.g., abide_AAL116, abide_schaefer100)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--repeat", type=int, default=5,
                       help="Number of repeated runs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    
    # System parameters
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Model architecture parameters
    parser.add_argument("--model", default="GCN",
                       choices=["GCN", "GPS"],
                       help="Model type")
    parser.add_argument("--num_layers", type=int, default=3,
                       help="Number of model layers")
    parser.add_argument("--dropout", type=float, default=0.5,
                       help="Dropout rate")
    parser.add_argument("--hidden_dim", type=int, default=64,
                       help="Hidden dimension for GCN")
    parser.add_argument("--channels", type=int, default=64,
                       help="Channels for GPS")
    parser.add_argument("--pooling", default="mean",
                       choices=["mean", "sum", "max"],
                       help="Global pooling function")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="Threshold for adjacency matrix conversion (GCN and GPS)")
    
    # GPS specific parameters
    parser.add_argument("--pe_dim", type=int, default=32,
                       help="Positional encoding dimension")
    parser.add_argument("--walk_length", type=int, default=8,
                       help="Walk length for RRWP positional encoding")
    parser.add_argument("--use_positional_encoding", action="store_true", default=True,
                       help="Use positional encoding for GPS")
    parser.add_argument("--attn_type", default="multihead",
                       choices=["multihead", "performer"],
                       help="Attention type for GPS")
    parser.add_argument("--heads", type=int, default=4,
                       help="Number of attention heads for GPS")
    
    args = parser.parse_args()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Create config with dynamic parameters
    cfg = DirectConfig()
    
    # Apply command line arguments
    cfg.set_dataset(args.dataset)
    cfg.model.name = args.model  # Use selected model (GCN or GPS)
    cfg.training.epochs = args.epochs
    cfg.repeat_time = args.repeat
    cfg.dataset.batch_size = args.batch_size
    cfg.dataset.test_batch_size = args.batch_size
    cfg.dataset.val_batch_size = args.batch_size
    cfg.optimizer[0].lr = args.lr
    cfg.optimizer[0].weight_decay = args.weight_decay
    cfg.seed = args.seed
    
    # Model architecture parameters
    cfg.model.num_layers = args.num_layers
    cfg.model.dropout = args.dropout
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.channels = args.channels
    cfg.model.pooling = args.pooling
    cfg.model.threshold = args.threshold
    
    # GPS specific parameters
    cfg.model.pe_dim = args.pe_dim
    cfg.model.walk_length = args.walk_length
    cfg.model.use_positional_encoding = args.use_positional_encoding
    cfg.model.attn_type = args.attn_type
    cfg.model.heads = args.heads
    
    print("Running {} with:".format(cfg.model.name))
    print("  Model: {}".format(cfg.model.name))
    print("  Dataset: {}".format(cfg.dataset.name.upper()))
    print("  Epochs: {}".format(cfg.training.epochs))
    print("  Repeat: {}".format(cfg.repeat_time))
    print("  Batch size: {}".format(cfg.dataset.batch_size))
    print("  Learning rate: {}".format(cfg.optimizer[0].lr))
    print("  Weight decay: {}".format(cfg.optimizer[0].weight_decay))
    if hasattr(cfg.model, 'pos_encoding'):
        print("  Positional encoding: {}".format(cfg.model.pos_encoding))
    elif hasattr(cfg.model, 'use_positional_encoding'):
        print("  Positional encoding: {}".format(cfg.model.use_positional_encoding))
    print("  GPU: {}".format(args.gpu))
    print("  Seed: {}".format(cfg.seed))
    
    # Set seed for reproducibility
    base_seed = cfg.get('seed', 42)
    deterministic = cfg.get('deterministic', True)
    
    print(" Reproducibility enabled with base_seed={}, deterministic={}".format(base_seed, deterministic))
    print(" Will run {} experiments with seeds: {}".format(cfg.repeat_time, [base_seed + i * 1000 for i in range(cfg.repeat_time)]))
    
    # Collect metrics across all runs
    all_test_aucs = []
    all_test_sensitivities = []
    all_test_specificities = []
    all_test_accuracies = []
    
    for run_idx in range(cfg.repeat_time):
        # Use 1000 offset approach for different random states
        run_seed = base_seed + run_idx * 1000  # Seeds: 42, 1042, 2042, 3042, 4042 for 5 runs
        set_seed(run_seed, deterministic)

        print("\nStarting run {}/{} with seed {}...".format(run_idx + 1, cfg.repeat_time, run_seed))
        final_metrics = model_training(cfg)
        
        # Collect metrics
        if final_metrics:
            all_test_aucs.append(final_metrics['test_auc'])
            all_test_sensitivities.append(final_metrics['test_sensitivity'])
            all_test_specificities.append(final_metrics['test_specificity'])
            all_test_accuracies.append(final_metrics['test_accuracy'])
            print("Run {} completed: AUC={:.4f}".format(run_idx + 1, final_metrics['test_auc']))
        else:
            print("Run {} failed!".format(run_idx + 1))

    
    # Calculate and display final statistics
    if all_test_aucs:
        print("\nSuccessfully completed {} out of {} runs".format(len(all_test_aucs), cfg.repeat_time))
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        print("Dataset: {}".format(cfg.dataset.name.upper()))
        print("Model: {}".format(cfg.model.name))
        print("Epochs: {}".format(cfg.training.epochs))
        print("Number of runs: {}".format(cfg.repeat_time))
        print("Seeds used: {}".format([base_seed + i * 1000 for i in range(cfg.repeat_time)]))
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
        
        # Display metrics based on dataset type
        if cfg.dataset.name.lower().startswith('ppmi'):
            print("Test Macro AUC:  {:.4f} ± {:.4f}".format(test_auc_mean, test_auc_std))
            print("Test Macro Sensitivity: {:.4f} ± {:.4f}".format(test_sen_mean, test_sen_std))
            print("Test Macro Specificity: {:.4f} ± {:.4f}".format(test_spe_mean, test_spe_std))
            print("Test Accuracy: {:.4f} ± {:.4f}".format(test_acc_mean, test_acc_std))
        else:
            print("Test ROC-AUC:  {:.4f} ± {:.4f}".format(test_auc_mean, test_auc_std))
            print("Test Sensitivity: {:.4f} ± {:.4f}".format(test_sen_mean, test_sen_std))
            print("Test Specificity: {:.4f} ± {:.4f}".format(test_spe_mean, test_spe_std))
            print("Test Accuracy: {:.4f} ± {:.4f}".format(test_acc_mean, test_acc_std))
        print("-"*80)
        print("Individual run results:")
        for i, (auc, sen, spe, acc) in enumerate(zip(all_test_aucs, all_test_sensitivities, all_test_specificities, all_test_accuracies)):
            print("  Seed {}: AUC={:.4f}, Sen={:.4f}, Spe={:.4f}, Acc={:.4f}".format(base_seed + i * 1000, auc, sen, spe, acc))
        print("="*80)
        
        # Save results to CSV
        csv_filename = save_results_to_csv(
            dataset_name=cfg.dataset.name,
            model_name=cfg.model.name,
            all_test_aucs=all_test_aucs,
            all_test_sensitivities=all_test_sensitivities,
            all_test_specificities=all_test_specificities,
            all_test_accuracies=all_test_accuracies,
            base_seed=base_seed,
            repeat_time=cfg.repeat_time
        )
    else:
        print("\n No runs completed successfully!")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
