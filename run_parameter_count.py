#!/usr/bin/env python3
"""
Simple runner script for parameter counting analysis.

Usage:
    python run_parameter_count.py
"""

from count_parameters import ParameterCounter
import pandas as pd

def quick_analysis():
    """Run a quick parameter analysis."""
    print("Running quick parameter analysis...")
    
    counter = ParameterCounter()
    
    # Analyze with default settings
    df = counter.analyze_all_models(node_size=90)
    
    # Print key results
    print("\n" + "="*80)
    print("QUICK PARAMETER COUNT RESULTS")
    print("="*80)
    
    # Show summary by rank
    for rank in [0, 1, 2, 4, 8]:
        rank_data = df[df['Rank'] == rank]
        print(f"\nRank {rank} Models:")
        for _, row in rank_data.iterrows():
            print(f"  {row['Model_Type']:12s}: {row['Parameters_Millions']:6.2f}M parameters")
    
    # Show scaling
    print(f"\nParameter Scaling (Rank 0 → Rank 8):")
    for model_type in df['Model_Type'].unique():
        rank0_params = df[(df['Model_Type'] == model_type) & (df['Rank'] == 0)]['Parameters_Millions'].iloc[0]
        rank8_params = df[(df['Model_Type'] == model_type) & (df['Rank'] == 8)]['Parameters_Millions'].iloc[0]
        scaling = rank8_params / rank0_params
        print(f"  {model_type:12s}: {scaling:.1f}x increase")
    
    return df

if __name__ == "__main__":
    df = quick_analysis()
    
    # Save results
    df.to_csv('quick_parameter_results.csv', index=False)
    print(f"\nResults saved to 'quick_parameter_results.csv'")
    print(f"Total models analyzed: {len(df)}")
