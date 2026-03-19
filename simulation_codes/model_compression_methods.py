#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:45:51 2025

@author: zoz
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy

def quantize_zero_aware(tensor, num_bits=8, min_nonzero=1e-4):  # Adjusted threshold
    q_tensor = tensor.clone()
    
    # Identify pre-existing zeros from pruning
    zero_mask = q_tensor == 0
    # Set small non-zero values to zero for zero-awareness, but preserve pruned zeros
    non_zero_mask = (q_tensor.abs() > min_nonzero) & (~zero_mask)
    
    if non_zero_mask.sum() == 0:
        return torch.quantize_per_tensor(q_tensor, 1.0, 0, torch.qint8)
    
    # Scale based on significant non-zeros
    max_val = q_tensor[non_zero_mask].abs().max()
    scale = max_val / 127.0  # Symmetric for qint8
    q_tensor[~non_zero_mask] = 0  # Zero only non-significant values
    
    # Quantize to qint8
    q_tensor_quant = torch.quantize_per_tensor(q_tensor, scale, 0, torch.qint8)
    
    return q_tensor_quant

def apply_zero_aware_quantization(model):
    quantized_model = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for name, module in quantized_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
                q_data = quantize_zero_aware(module.weight.data)
                module.weight = nn.Parameter(q_data, requires_grad=False)
    return quantized_model

def apply_pruning(model, amount=0.2):
    """Apply L1 unstructured pruning to all Conv2d and Linear layers"""
    pruned_model = copy.deepcopy(model)
    for _, module in pruned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return pruned_model

def make_sparse(model):
    sparse_model = copy.deepcopy(model)
    for _, module in sparse_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            sparse_weight = nn.Parameter(module.weight.to_sparse())
            module.register_parameter('weight', sparse_weight)
    return sparse_model

def make_dense(model):
    dense_model = copy.deepcopy(model)
    for _, module in dense_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight') and module.weight.data.is_sparse:
            dense_data = module.weight.data.to_dense()
            requires_grad = module.weight.requires_grad
            module.weight = nn.Parameter(dense_data, requires_grad=requires_grad)
    return dense_model

def make_dequant(model):
    dequant_model = copy.deepcopy(model)
    for _, module in dequant_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight') and module.weight.is_quantized:
            dequant_data = module.weight.dequantize()
            requires_grad = module.weight.requires_grad
            module.weight = nn.Parameter(dequant_data, requires_grad=requires_grad)
    return dequant_model
