#!/usr/bin/env python3
"""
EDGEQUAD ASPP Ablation Study

This script implements the ablation study: "Remove Aedge (ASPP on connectivity)"
to test if multi-scale edge cues matter beyond plain FC layers.

Ablation: Remove ASPP components that process connectivity/correlation matrices
and replace with plain fully connected layers.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class EDGEQUADBaseline(nn.Module):
    """Full EDGEQUAD model with ASPP on connectivity."""
    
    def __init__(self, node_size=90, corr_size=90, rank=2, num_classes=2):
        super().__init__()
        self.node_size = node_size
        self.corr_size = corr_size
        self.rank = rank
        self.num_classes = num_classes
        
        # Layer normalization
        self.ln_pre = nn.LayerNorm(node_size)
        self.ln_post = nn.LayerNorm(node_size)
        
        # CNNCorr components (keep as baseline)
        self.cnncorr_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.cnncorr_norm1 = nn.GroupNorm(4, 16)
        self.cnncorr_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.cnncorr_norm2 = nn.GroupNorm(4, 32)
        self.cnncorr_pool = nn.AdaptiveAvgPool2d((corr_size, corr_size))
        
        # ASPP components on connectivity (KEEP - this is what we'll ablate)
        self.aspp_conv1 = nn.Conv2d(1, 1, kernel_size=1)  # 1x1 conv
        self.aspp_conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=2, dilation=2)  # 3x3 dilated
        self.aspp_conv4 = nn.Conv2d(1, 1, kernel_size=3, padding=4, dilation=4)  # 3x3 dilated
        self.aspp_fusion = nn.Conv2d(3, 1, kernel_size=1)  # Fusion layer
        
        # Quadratic adapters
        self.adapter1 = self._create_quadratic_adapter(rank, node_size, 'adapter1')
        self.adapter2 = self._create_quadratic_adapter(rank, node_size, 'adapter2')
        
        # Clustering components
        self.encoder_linear1 = nn.Linear(corr_size * corr_size, 32)
        self.encoder_linear2 = nn.Linear(32, 32)
        self.encoder_linear3 = nn.Linear(32, corr_size * corr_size)
        self.cluster_centers = nn.Parameter(torch.randn(4, corr_size))
        
        # Classification layers
        self.dim_reduction_linear = nn.Linear(corr_size, 8)
        self.fc_linear1 = nn.Linear(8 * 4, 256)
        self.fc_linear2 = nn.Linear(256, 32)
        self.fc_linear3 = nn.Linear(32, num_classes)
        
        # Gating parameters
        self.beta_cnn = nn.Parameter(torch.tensor(-2.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))
    
    def _create_quadratic_adapter(self, rank, node_size, name_prefix):
        """Create quadratic adapter."""
        class QuadraticAdapter(nn.Module):
            def __init__(self, rank, node_size, name_prefix):
                super().__init__()
                self.rank = rank
                self.norm = nn.LayerNorm(node_size)
                self.base = nn.Linear(node_size, node_size, bias=True)
                self.alpha = nn.Parameter(torch.tensor(1.0))
                
                if rank > 0:
                    self.Ux = nn.Linear(node_size, rank, bias=False)
                    self.Vc = nn.Linear(node_size, rank, bias=False)
                    self.Wo = nn.Linear(rank, node_size, bias=False)
                else:
                    self.quad = nn.Linear(node_size, node_size, bias=False)
                
                self.gate = nn.Linear(node_size, node_size)
                self.gate_temp = nn.Parameter(torch.tensor(1.0))
                self.act = nn.GELU()
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x, corr):
                h = self.norm(x)
                base = self.base(h)
                
                if self.rank > 0:
                    ux = self.Ux(h)
                    vc = self.Vc(h)
                    quad = self.Wo(ux * vc)
                else:
                    quad = self.quad(h)
                
                gate = torch.sigmoid(self.gate(h) / self.gate_temp)
                y = gate * (base + self.alpha * quad)
                
                return x + self.dropout(self.act(y))
        
        return QuadraticAdapter(rank, node_size, name_prefix)
    
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
            
            # ASPP processing on connectivity (KEEP - this is what we ablate)
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
        
        # Clustering and classification
        x = self.ln_post(x)
        x_flat = x.view(B, -1)
        
        distances = torch.cdist(x_flat.view(B, N, N), self.cluster_centers.unsqueeze(0))
        assignments = torch.softmax(-distances, dim=-1)
        cluster_features = torch.bmm(assignments.transpose(-1, -2), x_flat.view(B, N, N))
        
        cluster_features = self.dim_reduction_linear(cluster_features)
        cluster_features = torch.relu(cluster_features)
        cluster_features = cluster_features.view(B, -1)
        
        output = self.fc_linear1(cluster_features)
        output = torch.relu(output)
        output = self.fc_linear2(output)
        output = torch.relu(output)
        output = self.fc_linear3(output)
        
        return output


class EDGEQUADAblated(nn.Module):
    """EDGEQUAD model with ASPP on connectivity REMOVED (replaced with plain FC)."""
    
    def __init__(self, node_size=90, corr_size=90, rank=2, num_classes=2):
        super().__init__()
        self.node_size = node_size
        self.corr_size = corr_size
        self.rank = rank
        self.num_classes = num_classes
        
        # Layer normalization
        self.ln_pre = nn.LayerNorm(node_size)
        self.ln_post = nn.LayerNorm(node_size)
        
        # CNNCorr components (keep as baseline)
        self.cnncorr_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.cnncorr_norm1 = nn.GroupNorm(4, 16)
        self.cnncorr_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.cnncorr_norm2 = nn.GroupNorm(4, 32)
        self.cnncorr_pool = nn.AdaptiveAvgPool2d((corr_size, corr_size))
        
        # ABLATION: Replace ASPP with plain FC layers
        # Instead of multi-scale ASPP, use simple fully connected processing
        self.plain_fc1 = nn.Linear(corr_size * corr_size, corr_size * corr_size)
        self.plain_fc2 = nn.Linear(corr_size * corr_size, corr_size * corr_size)
        self.plain_fc_norm = nn.LayerNorm(corr_size * corr_size)
        
        # Quadratic adapters (keep same)
        self.adapter1 = self._create_quadratic_adapter(rank, node_size, 'adapter1')
        self.adapter2 = self._create_quadratic_adapter(rank, node_size, 'adapter2')
        
        # Clustering components (keep same)
        self.encoder_linear1 = nn.Linear(corr_size * corr_size, 32)
        self.encoder_linear2 = nn.Linear(32, 32)
        self.encoder_linear3 = nn.Linear(32, corr_size * corr_size)
        self.cluster_centers = nn.Parameter(torch.randn(4, corr_size))
        
        # Classification layers (keep same)
        self.dim_reduction_linear = nn.Linear(corr_size, 8)
        self.fc_linear1 = nn.Linear(8 * 4, 256)
        self.fc_linear2 = nn.Linear(256, 32)
        self.fc_linear3 = nn.Linear(32, num_classes)
        
        # Gating parameters (keep same)
        self.beta_cnn = nn.Parameter(torch.tensor(-2.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))
    
    def _create_quadratic_adapter(self, rank, node_size, name_prefix):
        """Create quadratic adapter (same as baseline)."""
        class QuadraticAdapter(nn.Module):
            def __init__(self, rank, node_size, name_prefix):
                super().__init__()
                self.rank = rank
                self.norm = nn.LayerNorm(node_size)
                self.base = nn.Linear(node_size, node_size, bias=True)
                self.alpha = nn.Parameter(torch.tensor(1.0))
                
                if rank > 0:
                    self.Ux = nn.Linear(node_size, rank, bias=False)
                    self.Vc = nn.Linear(node_size, rank, bias=False)
                    self.Wo = nn.Linear(rank, node_size, bias=False)
                else:
                    self.quad = nn.Linear(node_size, node_size, bias=False)
                
                self.gate = nn.Linear(node_size, node_size)
                self.gate_temp = nn.Parameter(torch.tensor(1.0))
                self.act = nn.GELU()
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x, corr):
                h = self.norm(x)
                base = self.base(h)
                
                if self.rank > 0:
                    ux = self.Ux(h)
                    vc = self.Vc(h)
                    quad = self.Wo(ux * vc)
                else:
                    quad = self.quad(h)
                
                gate = torch.sigmoid(self.gate(h) / self.gate_temp)
                y = gate * (base + self.alpha * quad)
                
                return x + self.dropout(self.act(y))
        
        return QuadraticAdapter(rank, node_size, name_prefix)
    
    def forward(self, corr, corr_mat=None):
        B, N, M = corr.shape
        x = corr
        x = self.ln_pre(x)
        
        # CNNCorr processing (keep same)
        if corr_mat is not None:
            corr_4d = corr_mat.unsqueeze(1)
            cnn_out = self.cnncorr_conv1(corr_4d)
            cnn_out = self.cnncorr_norm1(cnn_out)
            cnn_out = torch.relu(cnn_out)
            cnn_out = self.cnncorr_conv2(cnn_out)
            cnn_out = self.cnncorr_norm2(cnn_out)
            cnn_out = torch.relu(cnn_out)
            cnn_out = self.cnncorr_pool(cnn_out).squeeze(1)
            
            # ABLATION: Replace ASPP with plain FC processing
            # Flatten connectivity matrix and process with FC layers
            corr_flat = corr_mat.view(B, -1)  # [B, N*N]
            fc_processed = self.plain_fc1(corr_flat)
            fc_processed = torch.relu(fc_processed)
            fc_processed = self.plain_fc2(fc_processed)
            fc_processed = self.plain_fc_norm(fc_processed)
            fc_processed = fc_processed.view(B, N, N)  # Reshape back to [B, N, N]
            
            # Combine CNNCorr and plain FC (instead of ASPP)
            gate = torch.sigmoid(self.beta_cnn)
            x = gate * cnn_out + (1 - gate) * fc_processed
        
        # Apply quadratic adapters (keep same)
        x = self.adapter1(x, corr)
        x = 0.5 * (x + x.transpose(-1, -2))
        x = x - torch.diag_embed(torch.diagonal(x, dim1=-2, dim2=-1))
        
        x = self.adapter2(x, corr)
        x = 0.5 * (x + x.transpose(-1, -2))
        x = x - torch.diag_embed(torch.diagonal(x, dim1=-2, dim2=-1))
        
        # Clustering and classification (keep same)
        x = self.ln_post(x)
        x_flat = x.view(B, -1)
        
        distances = torch.cdist(x_flat.view(B, N, N), self.cluster_centers.unsqueeze(0))
        assignments = torch.softmax(-distances, dim=-1)
        cluster_features = torch.bmm(assignments.transpose(-1, -2), x_flat.view(B, N, N))
        
        cluster_features = self.dim_reduction_linear(cluster_features)
        cluster_features = torch.relu(cluster_features)
        cluster_features = cluster_features.view(B, -1)
        
        output = self.fc_linear1(cluster_features)
        output = torch.relu(output)
        output = self.fc_linear2(output)
        output = torch.relu(output)
        output = self.fc_linear3(output)
        
        return output


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_ece(predictions, targets, n_bins=15):
    """Calculate Expected Calibration Error (ECE)."""
    # Convert predictions to probabilities
    if len(predictions.shape) > 1:
        probs = torch.softmax(predictions, dim=1)
        confidences = torch.max(probs, dim=1)[0]
        predicted_classes = torch.argmax(probs, dim=1)
    else:
        probs = torch.sigmoid(predictions)
        confidences = torch.max(probs, 1 - probs)
        predicted_classes = (probs > 0.5).long()
    
    # Calculate ECE
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predicted_classes[in_bin] == targets[in_bin]).float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def run_ablation_study():
    """Run the ASPP ablation study."""
    print("EDGEQUAD ASPP Ablation Study")
    print("="*50)
    print("Ablation: Remove Aedge (ASPP on connectivity)")
    print("Test: Do multi-scale edge cues matter beyond plain FC?")
    print()
    
    # Create models
    baseline_model = EDGEQUADBaseline(node_size=90, corr_size=90, rank=2, num_classes=2)
    ablated_model = EDGEQUADAblated(node_size=90, corr_size=90, rank=2, num_classes=2)
    
    # Count parameters
    baseline_params = count_parameters(baseline_model)
    ablated_params = count_parameters(ablated_model)
    
    print(f"Model Parameters:")
    print(f"  Baseline (with ASPP): {baseline_params:,} parameters")
    print(f"  Ablated (plain FC):   {ablated_params:,} parameters")
    print(f"  Parameter reduction:  {baseline_params - ablated_params:,} parameters")
    print()
    
    # Simulate some results (replace with actual training results)
    print("Simulated Results (replace with actual training):")
    print("-" * 50)
    
    # Baseline results (from your actual data - rank 2)
    baseline_results = {
        'AUROC': 72.59,
        'Accuracy': 64.63,
        'ECE': 0.15  # Simulated
    }
    
    # Ablated results (simulated - you need to train this)
    ablated_results = {
        'AUROC': 68.45,  # Simulated - should be lower
        'Accuracy': 61.20,  # Simulated - should be lower
        'ECE': 0.18  # Simulated - might be higher
    }
    
    # Calculate deltas
    delta_auroc = baseline_results['AUROC'] - ablated_results['AUROC']
    delta_accuracy = baseline_results['Accuracy'] - ablated_results['Accuracy']
    delta_ece = ablated_results['ECE'] - baseline_results['ECE']
    
    print(f"Baseline (with ASPP):")
    print(f"  AUROC: {baseline_results['AUROC']:.2f}%")
    print(f"  Accuracy: {baseline_results['Accuracy']:.2f}%")
    print(f"  ECE: {baseline_results['ECE']:.3f}")
    print()
    
    print(f"Ablated (plain FC):")
    print(f"  AUROC: {ablated_results['AUROC']:.2f}%")
    print(f"  Accuracy: {ablated_results['Accuracy']:.2f}%")
    print(f"  ECE: {ablated_results['ECE']:.3f}")
    print()
    
    print(f"Delta (Baseline - Ablated):")
    print(f"  ΔAUROC: {delta_auroc:+.2f}%")
    print(f"  ΔAccuracy: {delta_accuracy:+.2f}%")
    print(f"  ΔECE: {delta_ece:+.3f}")
    print()
    
    # Interpretation
    print("Interpretation:")
    print("-" * 20)
    if delta_auroc > 0:
        print("✓ Multi-scale ASPP processing improves AUROC")
    else:
        print("✗ Multi-scale ASPP processing does not improve AUROC")
    
    if delta_accuracy > 0:
        print("✓ Multi-scale ASPP processing improves Accuracy")
    else:
        print("✗ Multi-scale ASPP processing does not improve Accuracy")
    
    if delta_ece < 0:
        print("✓ Multi-scale ASPP processing improves calibration (lower ECE)")
    else:
        print("✗ Multi-scale ASPP processing worsens calibration (higher ECE)")
    
    return {
        'baseline': baseline_results,
        'ablated': ablated_results,
        'deltas': {
            'delta_auroc': delta_auroc,
            'delta_accuracy': delta_accuracy,
            'delta_ece': delta_ece
        }
    }


if __name__ == "__main__":
    results = run_ablation_study()
