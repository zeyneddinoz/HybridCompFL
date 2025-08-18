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



def quantize_zero_aware(tensor, num_bits=8, min_nonzero=0.01):
    qmax = 2**(num_bits - 1) - 1
    q_tensor = tensor.clone()
    
    # Preserve original zeros from pruning
    non_zero_mask = q_tensor.abs() > min_nonzero  # New threshold
    non_zero_vals = q_tensor[non_zero_mask]
    
    if non_zero_vals.numel() == 0:
        return q_tensor
        
    # Calculate scale based on preserved weights
    max_val = non_zero_vals.abs().max()
    scale = max_val / qmax
    
    # Quantize only sufficiently large weights
    quantized = torch.clamp(torch.round(non_zero_vals/scale), -qmax, qmax)
    q_tensor[non_zero_mask] = quantized * scale
    
    return q_tensor



def apply_zero_aware_quantization(model):
    quantized_model = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            if 'weight' in name:
                param.data = quantize_zero_aware(param.data)
    return quantized_model



def apply_pruning(model, amount=0.2):
    """Apply L1 unstructured pruning to all Conv2d and Linear layers"""
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Apply pruning and make it permanent
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model