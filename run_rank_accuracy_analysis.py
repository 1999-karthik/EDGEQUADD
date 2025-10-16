#!/usr/bin/env python3
"""
Simple runner for QUETT rank vs accuracy analysis.

This script reads your actual QUETT results and creates rank vs accuracy curves.
"""

from plot_rank_vs_accuracy import QUETTResultsAnalyzer
import pandas as pd

def run_rank_accuracy_analysis():
    """Run rank vs accuracy analysis on QUETT results."""
    print("QUETT Rank vs Accuracy Analysis")
    print("="*50)
    
    analyzer = QUETTResultsAnalyzer()
    
    # Parse results from your CSV file
    csv_file = 'QUETT/result/abide_AAL116.csv'
    results = analyzer.parse_quett_results(csv_file)
    
    if not results:
        print("No results found in the CSV file!")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    df = df.sort_values('rank')
    
    print(f"\nFound results for ranks: {sorted(df['rank'].unique())}")
    
    # Quick summary
    print(f"\nQuick Summary:")
    print("-" * 40)
    for _, row in df.iterrows():
        print(f"Rank {row['rank']:2d}: Acc={row['acc']:5.1f}%, ROC={row['roc']:5.1f}%, Sen={row['sen']:5.1f}%, Spec={row['spec']:5.1f}%")
    
    # Find best performing rank
    best_acc_idx = df['acc'].idxmax()
    best_acc_rank = df.loc[best_acc_idx, 'rank']
    best_acc_value = df.loc[best_acc_idx, 'acc']
    
    best_roc_idx = df['roc'].idxmax()
    best_roc_rank = df.loc[best_roc_idx, 'rank']
    best_roc_value = df.loc[best_roc_idx, 'roc']
    
    print(f"\nBest Performance:")
    print(f"  Best Accuracy: Rank {best_acc_rank} ({best_acc_value:.1f}%)")
    print(f"  Best ROC-AUC: Rank {best_roc_rank} ({best_roc_value:.1f}%)")
    
    # Create the plots
    print(f"\nCreating rank vs accuracy curves...")
    df_plots = analyzer.create_rank_vs_accuracy_plots('quett_rank_vs_accuracy_curves.png')
    
    # Create detailed summary
    analyzer.create_performance_summary_table(df)
    
    # Save results
    analyzer.save_results(df, 'quett_rank_analysis_results.csv')
    
    return df

if __name__ == "__main__":
    df = run_rank_accuracy_analysis()
    
    if df is not None:
        print(f"\nAnalysis complete!")
        print(f"Results saved to:")
        print(f"  - quett_rank_vs_accuracy_curves.png")
        print(f"  - quett_rank_analysis_results.csv")
