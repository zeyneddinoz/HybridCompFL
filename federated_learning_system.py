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
    global_rounds: int = 10,  # Standardized naming
    participation_rate: float = 0.3,
    local_epochs: int = 1,
    optimizer_name: str = 'Adam',  # Added for clarity
    learning_rate: float = 0.01,   # Added to pass LR
    test_loader: DataLoader = None
) -> nn.Module:
    
    device = next(global_model.parameters()).device
    num_selected = int(len(capable_devices) * participation_rate)
    aggregator = Aggregator()
    aggregation_fn = getattr(aggregator, f'fed_{aggregation_method.lower()}')
    
    capable_devices = sorted(capable_devices)
    print(f"The devices which are capable to train global model are: {capable_devices}")

    for round_num in range(global_rounds):
        print(f"\nGlobal Round {round_num+1}/{global_rounds}")
        
        # Select devices and collect updates
        selected = sorted(random.sample(capable_devices, num_selected))
        print(f"Participated devices: {selected}")  # Added logging of participating devices
        
        local_updates = []
        
        for dev_idx in selected:
            local_model = type(global_model)()
            local_model.load_state_dict(global_model.state_dict())
            local_model.to(device)
            
            # Use specified optimizer and LR
            if optimizer_name == 'Adam':
                optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)
            elif optimizer_name == 'SGD':
                optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
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
    learning_rate: float = 0.01,
    RESIZE_SIZE_FOR_VGG16: int = 224,
    alpha: float = 0.5
):
    
        
    device_loaders, test_loader, input_channels = prepare_datasets(dataset_name, 
                                                                   total_num_of_devices, 
                                                                   batch_size, 
                                                                   RESIZE_SIZE_FOR_VGG16,
                                                                   alpha)
    model = get_model(model_name, input_channels=input_channels)
    
    num_capable_devices = int(percentage_of_capable_devices * total_num_of_devices)
    capable_devices = random.sample(range(total_num_of_devices), num_capable_devices)  # Randomly select capable devices
    
    trained_model = federated_train(
        global_model=model,
        device_loaders=device_loaders,
        capable_devices=capable_devices,
        aggregation_method=aggregation_method,
        global_rounds=global_round,  # Standardized naming
        participation_rate=participation_rate, 
        local_epochs=local_epochs,
        optimizer_name=optimizer,    # Pass optimizer name
        learning_rate=learning_rate, # Pass LR
        test_loader=test_loader
    )
    
    return trained_model, test_loader