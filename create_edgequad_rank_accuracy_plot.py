#!/usr/bin/env python3
"""
Create EDGEQUAD Rank vs Accuracy Plot

This script creates a clean rank vs accuracy plot for EDGEQUAD models,
ensuring all references use "EDGEQUAD" instead of any other naming.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_edgequad_results(csv_file):
    """Parse EDGEQUAD results from CSV file."""
    results = []
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Parse the result line
        try:
            pairs = line.split(',')
            result = {}
            
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Parse different types of values
                    if key in ['roc', 'acc', 'sen', 'spec']:
                        # Extract mean value (before ±)
                        mean_val = float(value.split('±')[0].strip())
                        result[key] = mean_val
                    elif key in ['rank', 'layers', 'epochs', 'batch_size', 'cluster_num', 'cnncorr_base_ch', 'atrous_rate']:
                        result[key] = int(value)
                    elif key in ['base_lr', 'adapter_lr', 'wd', 'dropout', 'droppath']:
                        result[key] = float(value)
                    elif key in ['pooling']:
                        result[key] = value.lower() == 'true'
                    elif key in ['activation']:
                        result[key] = value
                    elif key in ['seeds', 'runs']:
                        # Handle list-like values
                        if '[' in value and ']' in value:
                            numbers = re.findall(r'\d+', value)
                            result[key] = [int(n) for n in numbers]
                        else:
                            result[key] = int(value)
                    elif key in ['cnncorr_aspp_rates', 'quadratic_aspp_rates']:
                        # Handle list values like [1, 2, 4]
                        if '[' in value and ']' in value:
                            numbers = re.findall(r'\d+', value)
                            result[key] = [int(n) for n in numbers]
                        else:
                            result[key] = value
                    else:
                        result[key] = value
            
            results.append(result)
            
        except Exception as e:
            print(f"Error parsing line: {line}")
            print(f"Error: {e}")
            continue
    
    return results

def create_edgequad_rank_accuracy_plot(csv_file, save_path=None):
    """Create EDGEQUAD rank vs accuracy plot."""
    
    # Parse results
    results = parse_edgequad_results(csv_file)
    if not results:
        print("No results found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('rank')
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Rank vs Accuracy
    ax1.plot(df['rank'], df['acc'], marker='o', linewidth=3, markersize=10, 
            color='blue', label='Accuracy')
    ax1.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('EDGEQUAD: Rank vs Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['rank'])
    
    # Add value labels on points
    for i, (rank, acc) in enumerate(zip(df['rank'], df['acc'])):
        ax1.annotate(f'{acc:.1f}%', (rank, acc), 
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=10, fontweight='bold')
    
    # Plot 2: Rank vs ROC-AUC
    ax2.plot(df['rank'], df['roc'], marker='s', linewidth=3, markersize=10, 
            color='red', label='ROC-AUC')
    ax2.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ROC-AUC (%)', fontsize=12, fontweight='bold')
    ax2.set_title('EDGEQUAD: Rank vs ROC-AUC', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['rank'])
    
    # Add value labels on points
    for i, (rank, roc) in enumerate(zip(df['rank'], df['roc'])):
        ax2.annotate(f'{roc:.1f}%', (rank, roc), 
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=10, fontweight='bold')
    
    # Plot 3: Rank vs Sensitivity and Specificity
    ax3.plot(df['rank'], df['sen'], marker='^', linewidth=3, markersize=10, 
            color='green', label='Sensitivity')
    ax3.plot(df['rank'], df['spec'], marker='v', linewidth=3, markersize=10, 
            color='orange', label='Specificity')
    ax3.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax3.set_title('EDGEQUAD: Rank vs Sensitivity & Specificity', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(df['rank'])
    
    # Plot 4: All metrics together
    ax4.plot(df['rank'], df['acc'], marker='o', linewidth=2, markersize=8, 
            color='blue', label='Accuracy')
    ax4.plot(df['rank'], df['roc'], marker='s', linewidth=2, markersize=8, 
            color='red', label='ROC-AUC')
    ax4.plot(df['rank'], df['sen'], marker='^', linewidth=2, markersize=8, 
            color='green', label='Sensitivity')
    ax4.plot(df['rank'], df['spec'], marker='v', linewidth=2, markersize=8, 
            color='orange', label='Specificity')
    ax4.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax4.set_title('EDGEQUAD: All Metrics vs Rank', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(df['rank'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"EDGEQUAD rank vs accuracy plot saved to {save_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("EDGEQUAD PERFORMANCE SUMMARY BY RANK")
    print("="*80)
    
    for _, row in df.iterrows():
        print(f"Rank {row['rank']:2d}: Acc={row['acc']:5.1f}%, ROC={row['roc']:5.1f}%, Sen={row['sen']:5.1f}%, Spec={row['spec']:5.1f}%")
    
    # Find best performing rank
    best_acc_idx = df['acc'].idxmax()
    best_acc_rank = df.loc[best_acc_idx, 'rank']
    best_acc_value = df.loc[best_acc_idx, 'acc']
    
    best_roc_idx = df['roc'].idxmax()
    best_roc_rank = df.loc[best_roc_idx, 'rank']
    best_roc_value = df.loc[best_roc_idx, 'roc']
    
    print(f"\nBest EDGEQUAD Performance:")
    print(f"  Best Accuracy: Rank {best_acc_rank} ({best_acc_value:.1f}%)")
    print(f"  Best ROC-AUC: Rank {best_roc_rank} ({best_roc_value:.1f}%)")
    
    return df

def main():
    """Main function to create EDGEQUAD rank vs accuracy plot."""
    print("Creating EDGEQUAD Rank vs Accuracy Plot")
    print("="*50)
    
    # Create the plot
    csv_file = 'QUETT/result/abide_AAL116.csv'
    df = create_edgequad_rank_accuracy_plot(csv_file, 'edgequad_rank_vs_accuracy_plot.png')
    
    if df is not None:
        # Save results to CSV
        df.to_csv('edgequad_rank_accuracy_results.csv', index=False)
        print(f"\nResults saved to 'edgequad_rank_accuracy_results.csv'")
        print(f"Analyzed {len(df)} EDGEQUAD configurations")

if __name__ == "__main__":
    main()