#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Tuning Script for QUETT on ABIDE Dataset

This script performs systematic hyperparameter tuning across multiple dimensions:
1. Learning rates (base_lr, adapter_lr)
2. Model architecture (layers, dropout, rank)
3. CNN correlation parameters (base channels, ASPP rates)
4. Quadratic parameters (ASPP rates, atrous rate)
5. Training parameters (weight decay, droppath)

Results are saved to CSV files for analysis.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from itertools import product
import argparse
from datetime import datetime

def run_experiment(args_dict, dataset='abide_AAL116', runs=5, base_seed=0):
    """
    Run a single experiment with given hyperparameters
    
    Args:
        args_dict: Dictionary of hyperparameters
        dataset: Dataset name
        runs: Number of runs
        base_seed: Base seed for reproducibility
    
    Returns:
        Dictionary with results
    """
    
    # Build command
    cmd = [
        'python', 'main.py',
        '--dataset', dataset,
        '--runs', str(runs),
        '--seed', str(base_seed),
        '--epochs', '200',
        '--batch_size', '16'
    ]
    
    # Add hyperparameters
    for key, value in args_dict.items():
        if isinstance(value, list):
            cmd.extend([f'--{key}'] + [str(v) for v in value])
        elif isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/pscratch/sd/s/saik1999/brain_Networks/QUETT')
        
        if result.returncode != 0:
            print(f"Error running experiment: {result.stderr}")
            return None
            
        # Parse results from stdout
        output_lines = result.stdout.strip().split('\n')
        
        # Look for the final results line
        for line in reversed(output_lines):
            if 'roc_auc ± std:' in line and 'mean ± std:' in line:
                # Parse the results
                parts = line.split(',')
                roc_part = parts[0].split(':')[1].strip()
                acc_part = parts[1].split(':')[1].strip()
                
                roc_mean = float(roc_part.split('±')[0].strip().replace('%', ''))
                roc_std = float(roc_part.split('±')[1].strip().replace('%', ''))
                acc_mean = float(acc_part.split('±')[0].strip().replace('%', ''))
                acc_std = float(acc_part.split('±')[1].strip().replace('%', ''))
                
                return {
                    'roc_mean': roc_mean,
                    'roc_std': roc_std,
                    'acc_mean': acc_mean,
                    'acc_std': acc_std,
                    'success': True
                }
        
        print("Could not parse results from output")
        return None
        
    except Exception as e:
        print(f"Exception running experiment: {e}")
        return None

def define_hyperparameter_grid():
    """
    Define the hyperparameter search grid
    """
    
    # Core hyperparameters to tune
    hyperparams = {
        # Learning rates
        'base_lr': [0.0001, 0.0005, 0.001],
        'adapter_lr': [0.0003, 0.0006, 0.001],
        'weight_decay': [0.0001, 0.0005, 0.001],
        
        # Model architecture
        'layers': [2, 3, 4],
        'dropout': [0.1, 0.2, 0.3],
        'rank': [2, 3, 4, 6],
        
        # CNN correlation parameters
        'cnncorr_base_ch': [16, 32, 64],
        'cnncorr_aspp_rates': [[1,2,4], [1,2,4,8], [1,3,6]],
        
        # Quadratic parameters
        'quadratic_aspp_rates': [[1,2,4], [1,2,4,8], [1,3,6]],
        'atrous_rate': [1, 2, 3],
        
        # Training parameters
        'droppath': [0.05, 0.1, 0.15],
    }
    
    return hyperparams

def create_search_strategies():
    """
    Create different search strategies for efficient hyperparameter tuning
    """
    
    strategies = {
        'coarse_grid': {
            'base_lr': [0.0001, 0.001],
            'adapter_lr': [0.0003, 0.001],
            'weight_decay': [0.0001, 0.001],
            'layers': [2, 3],
            'dropout': [0.1, 0.3],
            'rank': [2, 4],
            'cnncorr_base_ch': [16, 64],
            'cnncorr_aspp_rates': [[1,2,4], [1,3,6]],
            'quadratic_aspp_rates': [[1,2,4], [1,3,6]],
            'atrous_rate': [1, 3],
            'droppath': [0.05, 0.15],
        },
        
        'learning_rate_focus': {
            'base_lr': [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002],
            'adapter_lr': [0.0001, 0.0003, 0.0006, 0.001, 0.002],
            'weight_decay': [0.0001, 0.0005, 0.001],
            'layers': [3],  # Fixed
            'dropout': [0.2],  # Fixed
            'rank': [3],  # Fixed
            'cnncorr_base_ch': [32],  # Fixed
            'cnncorr_aspp_rates': [[1,2,4]],  # Fixed
            'quadratic_aspp_rates': [[1,2,4]],  # Fixed
            'atrous_rate': [2],  # Fixed
            'droppath': [0.05],  # Fixed
        },
        
        'architecture_focus': {
            'base_lr': [0.0001],  # Fixed
            'adapter_lr': [0.0006],  # Fixed
            'weight_decay': [0.0001],  # Fixed
            'layers': [1, 2, 3, 4, 5],
            'dropout': [0.1, 0.15, 0.2, 0.25, 0.3],
            'rank': [1, 2, 3, 4, 6, 8],
            'cnncorr_base_ch': [16, 24, 32, 48, 64],
            'cnncorr_aspp_rates': [[1,2], [1,2,4], [1,2,4,8], [1,3,6], [1,3,6,9]],
            'quadratic_aspp_rates': [[1,2], [1,2,4], [1,2,4,8], [1,3,6], [1,3,6,9]],
            'atrous_rate': [1, 2, 3, 4],
            'droppath': [0.0, 0.05, 0.1, 0.15, 0.2],
        }
    }
    
    return strategies

def run_hyperparameter_tuning(strategy_name='coarse_grid', max_experiments=None, dataset='abide_AAL116'):
    """
    Run hyperparameter tuning with specified strategy
    
    Args:
        strategy_name: Name of the search strategy
        max_experiments: Maximum number of experiments to run (None for all)
        dataset: Dataset name
    """
    
    strategies = create_search_strategies()
    
    if strategy_name not in strategies:
        print(f"Unknown strategy: {strategy_name}")
        print(f"Available strategies: {list(strategies.keys())}")
        return
    
    hyperparams = strategies[strategy_name]
    
    # Generate all combinations
    param_names = list(hyperparams.keys())
    param_values = list(hyperparams.values())
    
    all_combinations = list(product(*param_values))
    
    if max_experiments:
        all_combinations = all_combinations[:max_experiments]
    
    print(f"Running {len(all_combinations)} experiments with strategy: {strategy_name}")
    print(f"Dataset: {dataset}")
    
    results = []
    
    for i, combination in enumerate(all_combinations):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(all_combinations)}")
        print(f"{'='*60}")
        
        # Create args dictionary
        args_dict = dict(zip(param_names, combination))
        
        # Print current hyperparameters
        for key, value in args_dict.items():
            print(f"{key}: {value}")
        
        # Run experiment
        result = run_experiment(args_dict, dataset=dataset, runs=5, base_seed=0)
        
        if result and result['success']:
            # Add hyperparameters to result
            result.update(args_dict)
            result['experiment_id'] = i + 1
            result['strategy'] = strategy_name
            result['dataset'] = dataset
            results.append(result)
            
            print(f"Results: ROC={result['roc_mean']:.2f}±{result['roc_std']:.2f}%, "
                  f"ACC={result['acc_mean']:.2f}±{result['acc_std']:.2f}%")
        else:
            print("Experiment failed!")
        
        # Save intermediate results
        if results:
            df = pd.DataFrame(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hyperparameter_tuning_{strategy_name}_{dataset}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"Intermediate results saved to: {filename}")
    
    # Final results
    if results:
        df = pd.DataFrame(results)
        
        # Sort by ROC AUC
        df_sorted = df.sort_values('roc_mean', ascending=False)
        
        print(f"\n{'='*60}")
        print("TOP 10 RESULTS (by ROC AUC)")
        print(f"{'='*60}")
        
        for i, (_, row) in enumerate(df_sorted.head(10).iterrows()):
            print(f"{i+1:2d}. ROC: {row['roc_mean']:.2f}±{row['roc_std']:.2f}%, "
                  f"ACC: {row['acc_mean']:.2f}±{row['acc_std']:.2f}%")
            print(f"    Params: layers={row['layers']}, dropout={row['dropout']}, "
                  f"rank={row['rank']}, base_lr={row['base_lr']}, adapter_lr={row['adapter_lr']}")
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"final_hyperparameter_tuning_{strategy_name}_{dataset}_{timestamp}.csv"
        df_sorted.to_csv(final_filename, index=False)
        print(f"\nFinal results saved to: {final_filename}")
        
        return df_sorted
    else:
        print("No successful experiments!")
        return None

def main():
    parser = argparse.ArgumentParser(description='QUETT Hyperparameter Tuning')
    parser.add_argument('--strategy', type=str, default='coarse_grid',
                       choices=['coarse_grid', 'learning_rate_focus', 'architecture_focus'],
                       help='Search strategy to use')
    parser.add_argument('--max_experiments', type=int, default=None,
                       help='Maximum number of experiments to run')
    parser.add_argument('--dataset', type=str, default='abide_AAL116',
                       help='Dataset to use for tuning')
    
    args = parser.parse_args()
    
    print("QUETT Hyperparameter Tuning")
    print("=" * 50)
    print(f"Strategy: {args.strategy}")
    print(f"Dataset: {args.dataset}")
    print(f"Max experiments: {args.max_experiments or 'All'}")
    
    # Run hyperparameter tuning
    results = run_hyperparameter_tuning(
        strategy_name=args.strategy,
        max_experiments=args.max_experiments,
        dataset=args.dataset
    )
    
    if results is not None:
        print(f"\nHyperparameter tuning completed successfully!")
        print(f"Best ROC AUC: {results.iloc[0]['roc_mean']:.2f}%")
    else:
        print("\nHyperparameter tuning failed!")

if __name__ == "__main__":
    main()
