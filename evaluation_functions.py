#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:48:03 2025

@author: zoz
"""

import torch

from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np



def calculate_sparsity(model):
    total_weights = 0
    zero_weights = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_weights += param.numel()
            zero_weights += (param == 0).sum().item()
    return (zero_weights / total_weights) * 100 if total_weights > 0 else 0

def evaluate_model(model, test_loader, criterion, device='cpu'):
    model.eval()
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

def calculate_effective_size(model):
    size = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            size += param.numel() * 1  # 8 bits = 1 byte
        elif 'bias' in name:
            size += param.numel() * 4  # 32 bits = 4 bytes
    return size