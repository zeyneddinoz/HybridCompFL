#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:43:23 2025

@author: zoz
"""


import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from typing import List
import random

from model_aggregation_strategies import Aggregator
from model_architectures import get_model
from dataset_preparation import prepare_datasets


# Federated training loop 
def federated_train(
    global_model: nn.Module,
    device_loaders: List[DataLoader],
    capable_devices: List[int],
    aggregation_method: str,
    global_round: int = 10,
    participation_rate: float = 0.3,
    local_epochs: int = 1,
    test_loader: DataLoader = None
) -> nn.Module:
    
    device = next(global_model.parameters()).device
    num_selected = int(len(capable_devices) * participation_rate)
    aggregator = Aggregator()
    aggregation_fn = getattr(aggregator, f'fed_{aggregation_method.lower()}')

    for round_num in range(global_round):
        print(f"\nGlobal Round {round_num+1}/{global_round}")
        
        # Select devices and collect updates
        selected = random.sample(capable_devices, num_selected)
        local_updates = []
        
        for dev_idx in selected:
            local_model = type(global_model)()
            local_model.load_state_dict(global_model.state_dict())
            local_model.to(device)
            
            optimizer = optim.Adam(local_model.parameters())
            local_model.train()
            
            for _ in range(local_epochs):
                for X, y in device_loaders[dev_idx]:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(local_model(X), y)
                    loss.backward()
                    optimizer.step()
            
            local_updates.append(local_model.state_dict())
        
        # Aggregate updates
        global_weights = aggregation_fn(local_updates)
        global_model.load_state_dict(global_weights)
        
        # Evaluation
        if test_loader:
            global_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    correct += (global_model(X).argmax(1) == y).sum().item()
                    total += y.size(0)
            print(f"Test Accuracy: {100*correct/total:.2f}%")
    
    return global_model



def experiment_setting(
    model_name: str = 'VGG16',
    dataset_name: str = 'CIFAR10',
    aggregation_method: str = 'avg',
    global_round: int = 10,
    total_num_of_devices: int = 100, 
    participation_rate: float = 0.3,
    percentage_of_capable_devices: float = 0.2, 
    local_epochs: int = 1,
    batch_size: int = 32,
    optimizer: str = 'Adam',
    learning_rate: float = 0.01
):
    
        
    device_loaders, test_loader, input_channels = prepare_datasets(dataset_name, batch_size)
    model = get_model(model_name, input_channels=input_channels)
    
    # Determine capable device count based on dataset
    if dataset_name == 'CIFAR10':
        num_capable_devices = int(percentage_of_capable_devices * total_num_of_devices)  # 0.2 * 100: 20% of 100 devices
    else:  # MNIST or FashionMNIST
        num_capable_devices = 10  # 10% of 100 devices for simulation
   
    capable_devices = list(range(num_capable_devices))  # Capable devices
    
    trained_model = federated_train(
        global_model=model,
        device_loaders=device_loaders,
        capable_devices=capable_devices,
        aggregation_method=aggregation_method,
        global_round=global_round,
        participation_rate=participation_rate, 
        test_loader=test_loader,
        local_epochs=local_epochs
    )
    
    return trained_model, test_loader