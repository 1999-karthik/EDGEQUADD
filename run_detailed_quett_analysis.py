#!/usr/bin/env python3
"""
Simple runner for detailed QUETT parameter analysis.

This provides layer-by-layer parameter breakdown beyond model.parameters().
"""

from detailed_quett_parameter_analysis import DetailedQUETTAnalyzer
import pandas as pd

def run_detailed_analysis():
    """Run detailed QUETT parameter analysis."""
    print("Running detailed QUETT parameter analysis...")
    print("This provides layer-by-layer breakdown beyond model.parameters()")
    
    analyzer = DetailedQUETTAnalyzer()
    
    # Create models for different ranks
    models = analyzer.create_quett_variants(node_size=90, corr_size=90, layers=2, num_classes=2)
    
    print(f"\nCreated {len(models)} QUETT variants for ranks: {analyzer.ranks}")
    
    # Analyze each model
    analysis_results = []
    for model_name, model in models.items():
        result = analyzer.analyze_layer_parameters(model, model_name)
        analysis_results.append(result)
    
    # Create comparison table
    df = analyzer.create_comparison_table(analysis_results)
    
    # Print key results
    print("\n" + "="*80)
    print("QUETT PARAMETER SUMMARY BY RANK")
    print("="*80)
    
    for _, row in df.iterrows():
        print(f"Rank {row['Rank']:2d}: {row['Total_Parameters_M']:6.2f}M total parameters")
    
    # Show component breakdown for rank 0 and 8
    print(f"\nComponent Breakdown (Millions of Parameters):")
    print("-" * 60)
    print(f"{'Component':<20} {'Rank 0':<10} {'Rank 8':<10} {'Scaling':<10}")
    print("-" * 60)
    
    components = ['CNNCorr', 'ASPP', 'Quadratic_Adapters', 'Clustering', 'Normalization', 'Linear_Layers']
    
    rank0_row = df[df['Rank'] == 0].iloc[0]
    rank8_row = df[df['Rank'] == 8].iloc[0]
    
    for comp in components:
        rank0_val = rank0_row.get(f'{comp}_M', 0)
        rank8_val = rank8_row.get(f'{comp}_M', 0)
        scaling = rank8_val / rank0_val if rank0_val > 0 else 0
        print(f"{comp:<20} {rank0_val:<10.2f} {rank8_val:<10.2f} {scaling:<10.1f}x")
    
    # Save results
    df.to_csv('quett_detailed_comparison.csv', index=False)
    print(f"\nDetailed comparison saved to 'quett_detailed_comparison.csv'")
    
    return df, analysis_results

if __name__ == "__main__":
    df, results = run_detailed_analysis()
    
    print(f"\nAnalysis complete!")
    print(f"Total QUETT variants analyzed: {len(results)}")
    print(f"Ranks: {sorted(df['Rank'].unique())}")
