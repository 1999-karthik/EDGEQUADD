#!/usr/bin/env python3
"""
Detailed Parameter Analysis for QUETT Models

This script provides detailed parameter counting beyond model.parameters(),
showing layer-by-layer breakdown, component analysis, and rank-specific details.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import sys
import os
from pathlib import Path

# Add QUETT path
sys.path.append('QUETT')

class DetailedQUETTAnalyzer:
    """
    Detailed parameter analyzer for QUETT models with layer-by-layer breakdown.
    """
    
    def __init__(self):
        self.ranks = [0, 1, 2, 4, 8]
    
    def analyze_layer_parameters(self, model: nn.Module, model_name: str) -> Dict[str, Any]:
        """
        Analyze parameters layer by layer with detailed breakdown.
        """
        print(f"\nAnalyzing {model_name}...")
        
        layer_analysis = {}
        total_params = 0
        
        # Analyze each named module
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_params = sum(p.numel() for p in module.parameters())
                if module_params > 0:
                    layer_analysis[name] = {
                        'module_type': type(module).__name__,
                        'parameters': module_params,
                        'parameters_millions': module_params / 1e6,
                        'trainable_params': sum(p.numel() for p in module.parameters() if p.requires_grad),
                        'non_trainable_params': sum(p.numel() for p in module.parameters() if not p.requires_grad)
                    }
                    total_params += module_params
        
        # Group by component type
        component_analysis = self._group_by_component(layer_analysis)
        
        return {
            'model_name': model_name,
            'total_parameters': total_params,
            'total_parameters_millions': total_params / 1e6,
            'layer_analysis': layer_analysis,
            'component_analysis': component_analysis
        }
    
    def _group_by_component(self, layer_analysis: Dict) -> Dict[str, Dict]:
        """Group layers by component type."""
        components = {
            'CNNCorr': {'params': 0, 'layers': []},
            'ASPP': {'params': 0, 'layers': []},
            'Quadratic_Adapters': {'params': 0, 'layers': []},
            'Clustering': {'params': 0, 'layers': []},
            'Normalization': {'params': 0, 'layers': []},
            'Linear_Layers': {'params': 0, 'layers': []},
            'Other': {'params': 0, 'layers': []}
        }
        
        for layer_name, layer_info in layer_analysis.items():
            params = layer_info['parameters']
            
            # Categorize by layer name and type
            if 'cnncorr' in layer_name.lower():
                components['CNNCorr']['params'] += params
                components['CNNCorr']['layers'].append((layer_name, params))
            elif 'aspp' in layer_name.lower():
                components['ASPP']['params'] += params
                components['ASPP']['layers'].append((layer_name, params))
            elif 'adapter' in layer_name.lower() or 'quadratic' in layer_name.lower():
                components['Quadratic_Adapters']['params'] += params
                components['Quadratic_Adapters']['layers'].append((layer_name, params))
            elif 'dec' in layer_name.lower() or 'cluster' in layer_name.lower() or 'encoder' in layer_name.lower():
                components['Clustering']['params'] += params
                components['Clustering']['layers'].append((layer_name, params))
            elif 'norm' in layer_name.lower() or 'ln' in layer_name.lower():
                components['Normalization']['params'] += params
                components['Normalization']['layers'].append((layer_name, params))
            elif 'linear' in layer_name.lower() or 'fc' in layer_name.lower():
                components['Linear_Layers']['params'] += params
                components['Linear_Layers']['layers'].append((layer_name, params))
            else:
                components['Other']['params'] += params
                components['Other']['layers'].append((layer_name, params))
        
        return components
    
    def create_quett_variants(self, node_size: int = 90, corr_size: int = 90, 
                             layers: int = 2, num_classes: int = 2) -> Dict[str, nn.Module]:
        """Create QUETT variants with different ranks."""
        models = {}
        
        for rank in self.ranks:
            model_name = f'QUETT_Rank{rank}'
            models[model_name] = self._create_simplified_quett(rank, node_size, corr_size, num_classes)
        
        return models
    
    def _create_simplified_quett(self, rank: int, node_size: int, corr_size: int, num_classes: int) -> nn.Module:
        """Create a detailed QUETT-like model for analysis."""
        
        class DetailedQUETT(nn.Module):
            def __init__(self, rank, node_size, corr_size, num_classes):
                super().__init__()
                self.rank = rank
                self.node_size = node_size
                self.corr_size = corr_size
                self.num_classes = num_classes
                
                # Layer normalization components
                self.ln_pre = nn.LayerNorm(node_size, name='ln_pre')
                self.ln_post = nn.LayerNorm(node_size, name='ln_post')
                
                # CNNCorr components
                self.cnncorr_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, name='cnncorr_conv1')
                self.cnncorr_norm1 = nn.GroupNorm(4, 16, name='cnncorr_norm1')
                self.cnncorr_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, name='cnncorr_conv2')
                self.cnncorr_norm2 = nn.GroupNorm(4, 32, name='cnncorr_norm2')
                self.cnncorr_pool = nn.AdaptiveAvgPool2d((corr_size, corr_size), name='cnncorr_pool')
                
                # ASPP components
                self.aspp_conv1 = nn.Conv2d(1, 1, kernel_size=1, name='aspp_conv1')
                self.aspp_conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=2, dilation=2, name='aspp_conv2')
                self.aspp_conv4 = nn.Conv2d(1, 1, kernel_size=3, padding=4, dilation=4, name='aspp_conv4')
                self.aspp_fusion = nn.Conv2d(3, 1, kernel_size=1, name='aspp_fusion')
                
                # Quadratic adapters (2 layers)
                self.adapter1 = self._create_quadratic_adapter(rank, node_size, 'adapter1')
                self.adapter2 = self._create_quadratic_adapter(rank, node_size, 'adapter2')
                
                # Clustering/DEC components
                self.encoder_linear1 = nn.Linear(corr_size * corr_size, 32, name='encoder_linear1')
                self.encoder_linear2 = nn.Linear(32, 32, name='encoder_linear2')
                self.encoder_linear3 = nn.Linear(32, corr_size * corr_size, name='encoder_linear3')
                
                # Clustering centers (4 clusters)
                self.cluster_centers = nn.Parameter(torch.randn(4, corr_size), name='cluster_centers')
                
                # Final classification layers
                self.dim_reduction_linear = nn.Linear(corr_size, 8, name='dim_reduction_linear')
                self.fc_linear1 = nn.Linear(8 * 4, 256, name='fc_linear1')
                self.fc_linear2 = nn.Linear(256, 32, name='fc_linear2')
                self.fc_linear3 = nn.Linear(32, num_classes, name='fc_linear3')
                
                # Gating parameters
                self.beta_cnn = nn.Parameter(torch.tensor(-2.0), name='beta_cnn')
                self.alpha = nn.Parameter(torch.tensor(1.0), name='alpha')
            
            def _create_quadratic_adapter(self, rank, node_size, name_prefix):
                """Create a detailed quadratic adapter."""
                class DetailedQuadraticAdapter(nn.Module):
                    def __init__(self, rank, node_size, name_prefix):
                        super().__init__()
                        self.rank = rank
                        self.norm = nn.LayerNorm(node_size, name=f'{name_prefix}_norm')
                        self.base = nn.Linear(node_size, node_size, bias=True, name=f'{name_prefix}_base')
                        self.alpha = nn.Parameter(torch.tensor(1.0), name=f'{name_prefix}_alpha')
                        
                        # Quadratic term
                        if rank > 0:
                            self.Ux = nn.Linear(node_size, rank, bias=False, name=f'{name_prefix}_Ux')
                            self.Vc = nn.Linear(node_size, rank, bias=False, name=f'{name_prefix}_Vc')
                            self.Wo = nn.Linear(rank, node_size, bias=False, name=f'{name_prefix}_Wo')
                        else:
                            self.quad = nn.Linear(node_size, node_size, bias=False, name=f'{name_prefix}_quad')
                        
                        self.gate = nn.Linear(node_size, node_size, name=f'{name_prefix}_gate')
                        self.gate_temp = nn.Parameter(torch.tensor(1.0), name=f'{name_prefix}_gate_temp')
                        self.act = nn.GELU()
                        self.dropout = nn.Dropout(0.1, name=f'{name_prefix}_dropout')
                    
                    def forward(self, x, corr):
                        h = self.norm(x)
                        base = self.base(h)
                        
                        # Quadratic term
                        if self.rank > 0:
                            ux = self.Ux(h)
                            vc = self.Vc(h)
                            quad = self.Wo(ux * vc)
                        else:
                            quad = self.quad(h)
                        
                        # Gating
                        gate = torch.sigmoid(self.gate(h) / self.gate_temp)
                        y = gate * (base + self.alpha * quad)
                        
                        return x + self.dropout(self.act(y))
                
                return DetailedQuadraticAdapter(rank, node_size, name_prefix)
            
            def forward(self, corr, corr_mat=None):
                B, N, M = corr.shape
                
                x = corr
                x = self.ln_pre(x)
                
                # CNNCorr processing
                if corr_mat is not None:
                    corr_4d = corr_mat.unsqueeze(1)
                    cnn_out = self.cnncorr_conv1(corr_4d)
                    cnn_out = self.cnncorr_norm1(cnn_out)
                    cnn_out = torch.relu(cnn_out)
                    cnn_out = self.cnncorr_conv2(cnn_out)
                    cnn_out = self.cnncorr_norm2(cnn_out)
                    cnn_out = torch.relu(cnn_out)
                    cnn_out = self.cnncorr_pool(cnn_out).squeeze(1)
                    
                    # ASPP processing
                    aspp1 = self.aspp_conv1(corr_4d)
                    aspp2 = self.aspp_conv2(corr_4d)
                    aspp4 = self.aspp_conv4(corr_4d)
                    aspp_fused = self.aspp_fusion(torch.cat([aspp1, aspp2, aspp4], dim=1)).squeeze(1)
                    
                    # Combine CNNCorr and ASPP
                    gate = torch.sigmoid(self.beta_cnn)
                    x = gate * cnn_out + (1 - gate) * aspp_fused
                
                # Apply quadratic adapters
                x = self.adapter1(x, corr)
                x = 0.5 * (x + x.transpose(-1, -2))
                x = x - torch.diag_embed(torch.diagonal(x, dim1=-2, dim2=-1))
                
                x = self.adapter2(x, corr)
                x = 0.5 * (x + x.transpose(-1, -2))
                x = x - torch.diag_embed(torch.diagonal(x, dim1=-2, dim2=-1))
                
                # Clustering
                x = self.ln_post(x)
                x_flat = x.view(B, -1)
                
                # Simple clustering
                distances = torch.cdist(x_flat.view(B, N, N), self.cluster_centers.unsqueeze(0))
                assignments = torch.softmax(-distances, dim=-1)
                cluster_features = torch.bmm(assignments.transpose(-1, -2), x_flat.view(B, N, N))
                
                # Classification
                cluster_features = self.dim_reduction_linear(cluster_features)
                cluster_features = torch.relu(cluster_features)
                cluster_features = cluster_features.view(B, -1)
                
                output = self.fc_linear1(cluster_features)
                output = torch.relu(output)
                output = self.fc_linear2(output)
                output = torch.relu(output)
                output = self.fc_linear3(output)
                
                return output
        
        return DetailedQUETT(rank, node_size, corr_size, num_classes)
    
    def print_detailed_analysis(self, analysis_results: List[Dict]):
        """Print detailed analysis results."""
        print("\n" + "="*120)
        print("DETAILED QUETT PARAMETER ANALYSIS")
        print("="*120)
        
        for result in analysis_results:
            model_name = result['model_name']
            total_params = result['total_parameters_millions']
            
            print(f"\n{model_name} - Total: {total_params:.2f}M parameters")
            print("-" * 80)
            
            # Component breakdown
            component_analysis = result['component_analysis']
            for component, info in component_analysis.items():
                if info['params'] > 0:
                    params_millions = info['params'] / 1e6
                    percentage = (info['params'] / result['total_parameters']) * 100
                    print(f"\n{component}: {params_millions:.2f}M ({percentage:.1f}%)")
                    
                    # Show individual layers
                    for layer_name, layer_params in info['layers']:
                        layer_params_millions = layer_params / 1e6
                        print(f"  └─ {layer_name}: {layer_params_millions:.3f}M")
    
    def create_comparison_table(self, analysis_results: List[Dict]) -> pd.DataFrame:
        """Create a comparison table across ranks."""
        comparison_data = []
        
        for result in analysis_results:
            model_name = result['model_name']
            rank = int(model_name.split('_')[-1].replace('Rank', ''))
            
            row = {
                'Rank': rank,
                'Total_Parameters_M': result['total_parameters_millions'],
                'Total_Parameters': result['total_parameters']
            }
            
            # Add component breakdown
            component_analysis = result['component_analysis']
            for component, info in component_analysis.items():
                row[f'{component}_M'] = info['params'] / 1e6
                row[f'{component}_%'] = (info['params'] / result['total_parameters']) * 100
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Rank')
        
        return df
    
    def print_comparison_table(self, df: pd.DataFrame):
        """Print formatted comparison table."""
        print("\n" + "="*120)
        print("QUETT PARAMETER COMPARISON ACROSS RANKS")
        print("="*120)
        
        # Main comparison
        print("\nTotal Parameters by Rank:")
        print("-" * 40)
        for _, row in df.iterrows():
            print(f"Rank {row['Rank']:2d}: {row['Total_Parameters_M']:6.2f}M parameters")
        
        # Component comparison
        components = ['CNNCorr', 'ASPP', 'Quadratic_Adapters', 'Clustering', 'Normalization', 'Linear_Layers', 'Other']
        
        print(f"\nComponent Breakdown (Millions of Parameters):")
        print("-" * 80)
        print(f"{'Rank':<6}", end="")
        for comp in components:
            print(f"{comp[:12]:<12}", end="")
        print()
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"{row['Rank']:<6}", end="")
            for comp in components:
                value = row.get(f'{comp}_M', 0)
                print(f"{value:<12.2f}", end="")
            print()
        
        # Scaling analysis
        print(f"\nParameter Scaling Analysis:")
        print("-" * 50)
        for i in range(1, len(df)):
            prev_params = df.iloc[i-1]['Total_Parameters_M']
            curr_params = df.iloc[i]['Total_Parameters_M']
            rank_prev = df.iloc[i-1]['Rank']
            rank_curr = df.iloc[i]['Rank']
            
            if prev_params > 0:
                scaling_factor = curr_params / prev_params
                print(f"Rank {rank_prev} → Rank {rank_curr}: {scaling_factor:.2f}x increase")
    
    def save_detailed_results(self, analysis_results: List[Dict], df: pd.DataFrame):
        """Save detailed results to files."""
        # Save summary CSV
        df.to_csv('quett_detailed_parameter_comparison.csv', index=False)
        
        # Save detailed analysis
        with open('quett_detailed_layer_analysis.txt', 'w') as f:
            f.write("DETAILED QUETT PARAMETER ANALYSIS\n")
            f.write("="*50 + "\n\n")
            
            for result in analysis_results:
                f.write(f"{result['model_name']} - Total: {result['total_parameters_millions']:.2f}M parameters\n")
                f.write("-" * 50 + "\n")
                
                component_analysis = result['component_analysis']
                for component, info in component_analysis.items():
                    if info['params'] > 0:
                        params_millions = info['params'] / 1e6
                        percentage = (info['params'] / result['total_parameters']) * 100
                        f.write(f"\n{component}: {params_millions:.2f}M ({percentage:.1f}%)\n")
                        
                        for layer_name, layer_params in info['layers']:
                            layer_params_millions = layer_params / 1e6
                            f.write(f"  └─ {layer_name}: {layer_params_millions:.3f}M\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"\nDetailed results saved to:")
        print(f"  - quett_detailed_parameter_comparison.csv")
        print(f"  - quett_detailed_layer_analysis.txt")


def main():
    """Main function for detailed QUETT analysis."""
    print("Detailed QUETT Parameter Analysis")
    print("="*50)
    
    analyzer = DetailedQUETTAnalyzer()
    
    # Create models
    models = analyzer.create_quett_variants(node_size=90, corr_size=90, layers=2, num_classes=2)
    
    # Analyze each model
    analysis_results = []
    for model_name, model in models.items():
        result = analyzer.analyze_layer_parameters(model, model_name)
        analysis_results.append(result)
    
    # Print detailed analysis
    analyzer.print_detailed_analysis(analysis_results)
    
    # Create comparison table
    df = analyzer.create_comparison_table(analysis_results)
    analyzer.print_comparison_table(df)
    
    # Save results
    analyzer.save_detailed_results(analysis_results, df)
    
    print(f"\nDetailed analysis complete!")
    print(f"Analyzed {len(analysis_results)} QUETT variants")


if __name__ == "__main__":
    main()
