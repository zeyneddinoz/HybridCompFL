#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:48:03 2025

@author: zoz
"""


import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from model_compression_methods import make_dense, make_dequant



def calculate_sparsity(model):
    total_weights = 0
    zero_weights = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.data.is_sparse:
                total = param.numel()
                non_zeros = param.data._nnz()
            elif param.data.is_quantized:  # Updated check
                p = param.data.dequantize()
                total = p.numel()
                non_zeros = (p != 0).sum().item()
            else:
                total = param.numel()
                non_zeros = (param != 0).sum().item()
            total_weights += total
            zero_weights += total - non_zeros
    return (zero_weights / total_weights) * 100 if total_weights > 0 else 0

def evaluate_model(model, test_loader, criterion, device='cpu'):
    # If model is sparse or quantized, use dense/dequant version for eval
    if any(p.data.is_sparse for p in model.parameters()):
        model = make_dense(model)
    if any(p.data.is_quantized for p in model.parameters()):  # Updated check
        model = make_dequant(model)
    
    model.to(device)
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    return {
        'loss': total_loss / len(test_loader.dataset),
        'accuracy': (all_preds == all_targets).mean(),
        'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0)
    }

# Note: calculate_effective_size is removed; use actual file size instead