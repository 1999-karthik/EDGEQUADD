#!/usr/bin/env python3
"""
Rank vs Accuracy Curve Plotter for QUETT Models

This script reads the actual QUETT results and plots rank vs accuracy curves
along with other performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

class QUETTResultsAnalyzer:
    """
    Analyzes QUETT results and creates rank vs accuracy curves.
    """
    
    def __init__(self):
        self.results = []
    
    def parse_quett_results(self, csv_file: str):
        """
        Parse QUETT results from CSV file.
        
        Args:
            csv_file: Path to the CSV file with QUETT results
        """
        print(f"Reading QUETT results from {csv_file}...")
        
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse the result line
            result = self._parse_result_line(line)
            if result:
                self.results.append(result)
        
        print(f"Parsed {len(self.results)} results")
        return self.results
    
    def _parse_result_line(self, line: str) -> dict:
        """
        Parse a single result line from the CSV.
        
        Example line:
        roc:70.01 ± 0.00,acc:60.30 ± 0.00,sen:50.00 ± 0.00,spec:69.44 ± 0.00,seeds:[42],runs:1,epochs:200,batch_size:16,base_lr:0.0001,adapter_lr:0.0006,wd:0.0001,layers:3,activation:leaky_relu,dropout:0.2,pooling:True,cluster_num:4,droppath:0.05,rank:0,cnncorr_base_ch:32,cnncorr_aspp_rates:[1, 2, 4],quadratic_aspp_rates:[1, 2, 4],atrous_rate:2
        """
        try:
            # Split by comma and parse key-value pairs
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
                            # Extract numbers from [42] format
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
            
            return result
            
        except Exception as e:
            print(f"Error parsing line: {line}")
            print(f"Error: {e}")
            return None
    
    def create_rank_vs_accuracy_plots(self, save_path: str = None):
        """
        Create comprehensive rank vs accuracy plots.
        
        Args:
            save_path: Path to save the plots
        """
        if not self.results:
            print("No results to plot!")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.results)
        df = df.sort_values('rank')
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Rank vs Accuracy
        ax1.plot(df['rank'], df['acc'], marker='o', linewidth=3, markersize=10, 
                color='blue', label='Accuracy')
        ax1.set_xlabel('Rank', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('QUETT: Rank vs Accuracy', fontsize=14, fontweight='bold')
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
        ax2.set_title('QUETT: Rank vs ROC-AUC', fontsize=14, fontweight='bold')
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
        ax3.set_title('QUETT: Rank vs Sensitivity & Specificity', fontsize=14, fontweight='bold')
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
        ax4.set_title('QUETT: All Metrics vs Rank', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(df['rank'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        
        plt.show()
        
        return df
    
    def create_performance_summary_table(self, df: pd.DataFrame):
        """
        Create a summary table of performance metrics.
        
        Args:
            df: DataFrame with results
        """
        print("\n" + "="*80)
        print("QUETT PERFORMANCE SUMMARY BY RANK")
        print("="*80)
        
        # Create summary table
        summary_cols = ['rank', 'acc', 'roc', 'sen', 'spec']
        summary_df = df[summary_cols].copy()
        summary_df.columns = ['Rank', 'Accuracy (%)', 'ROC-AUC (%)', 'Sensitivity (%)', 'Specificity (%)']
        
        # Round to 2 decimal places
        for col in summary_df.columns[1:]:
            summary_df[col] = summary_df[col].round(2)
        
        print(summary_df.to_string(index=False))
        
        # Find best performing rank for each metric
        print(f"\n" + "="*80)
        print("BEST PERFORMING RANKS")
        print("="*80)
        
        best_acc_rank = df.loc[df['acc'].idxmax(), 'rank']
        best_acc_value = df['acc'].max()
        print(f"Best Accuracy: Rank {best_acc_rank} ({best_acc_value:.2f}%)")
        
        best_roc_rank = df.loc[df['roc'].idxmax(), 'rank']
        best_roc_value = df['roc'].max()
        print(f"Best ROC-AUC: Rank {best_roc_rank} ({best_roc_value:.2f}%)")
        
        best_sen_rank = df.loc[df['sen'].idxmax(), 'rank']
        best_sen_value = df['sen'].max()
        print(f"Best Sensitivity: Rank {best_sen_rank} ({best_sen_value:.2f}%)")
        
        best_spec_rank = df.loc[df['spec'].idxmax(), 'rank']
        best_spec_value = df['spec'].max()
        print(f"Best Specificity: Rank {best_spec_rank} ({best_spec_value:.2f}%)")
        
        # Performance trends
        print(f"\n" + "="*80)
        print("PERFORMANCE TRENDS")
        print("="*80)
        
        # Calculate trends (simple linear trend)
        ranks = df['rank'].values
        accs = df['acc'].values
        rocs = df['roc'].values
        
        # Simple trend analysis
        acc_trend = "increasing" if accs[-1] > accs[0] else "decreasing"
        roc_trend = "increasing" if rocs[-1] > rocs[0] else "decreasing"
        
        print(f"Accuracy trend (Rank 0 → Rank 8): {acc_trend}")
        print(f"ROC-AUC trend (Rank 0 → Rank 8): {roc_trend}")
        
        # Performance variation
        acc_std = df['acc'].std()
        roc_std = df['roc'].std()
        print(f"Accuracy standard deviation: {acc_std:.2f}%")
        print(f"ROC-AUC standard deviation: {roc_std:.2f}%")
    
    def save_results(self, df: pd.DataFrame, filename: str = 'quett_rank_analysis.csv'):
        """
        Save analysis results to CSV.
        
        Args:
            df: DataFrame with results
            filename: Output filename
        """
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")


def main():
    """Main function to run the rank vs accuracy analysis."""
    print("QUETT Rank vs Accuracy Analysis")
    print("="*50)
    
    # Initialize analyzer
    analyzer = QUETTResultsAnalyzer()
    
    # Parse results from CSV file
    csv_file = 'QUETT/result/abide_AAL116.csv'
    results = analyzer.parse_quett_results(csv_file)
    
    if not results:
        print("No results found!")
        return
    
    # Create plots
    df = analyzer.create_rank_vs_accuracy_plots('quett_rank_vs_accuracy_curves.png')
    
    # Create summary table
    analyzer.create_performance_summary_table(df)
    
    # Save results
    analyzer.save_results(df)
    
    print(f"\nAnalysis complete!")
    print(f"Analyzed {len(results)} QUETT configurations")


if __name__ == "__main__":
    main()
