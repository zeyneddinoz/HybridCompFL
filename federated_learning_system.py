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
# Removed: import matplotlib.pyplot as plt  (no longer needed here)

from model_aggregation_strategies import Aggregator
from model_architectures import get_model
from dataset_preparation import prepare_datasets

# Federated training loop 
def federated_train(
    global_model: nn.Module,
    device_loaders: List[DataLoader],
    capable_devices: List[int],
    aggregation_method: str,
    global_rounds: int = 10,
    participation_rate: float = 0.3,
    local_epochs: int = 1,
    optimizer_name: str = 'Adam',
    learning_rate: float = 0.01,
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
        print(f"Participated devices: {selected}")
        
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
    
    # Receive train_dataset along with other returns
    device_loaders, test_loader, input_channels, client_indices, train_dataset = prepare_datasets(
        dataset_name, 
        total_num_of_devices, 
        batch_size, 
        RESIZE_SIZE_FOR_VGG16,
        alpha
    )

    model = get_model(model_name, input_channels=input_channels)
    
    num_capable_devices = int(percentage_of_capable_devices * total_num_of_devices)
    capable_devices = random.sample(range(total_num_of_devices), num_capable_devices)
    
    # Print data distribution and create plots for capable devices
    labels = torch.tensor(train_dataset.targets)
    num_classes = len(torch.unique(labels))  # Define num_classes before use
    
    # NEW: Compute total class counts for the entire dataset
    total_class_counts = [(labels == c).sum().item() for c in range(num_classes)]
    
    # Define class names based on dataset
    if dataset_name == 'FashionMNIST':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_name == 'CIFAR10':
        class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    elif dataset_name == 'MNIST':
        class_names = [str(i) for i in range(10)]  # Digits 0-9
    else:
        class_names = [str(i) for i in range(num_classes)]  # Fallback for unknown datasets

    print("\nData Distribution for Capable Devices:")
    
    # NEW: Collect data distribution into a dictionary
    data_dist = {
        "class_names": class_names,
        "capable_devices": sorted(capable_devices),
        "distributions": {},
        "total_class_counts": total_class_counts  # NEW: Add total class counts
    }

    for dev_idx in sorted(capable_devices):  # Sort for consistent output
        subset_indices = client_indices[dev_idx]
        subset_labels = labels[subset_indices]
        class_counts = [(subset_labels == c).sum().item() for c in range(num_classes)]
        print(f"Capable Device {dev_idx} data distribution: {class_counts}")
        
        # NEW: Store in dict (use str key for JSON compatibility)
        data_dist["distributions"][str(dev_idx)] = class_counts
        
        # Removed: Individual plot code (subplots, bar plots, savefig, etc.) as we'll plot stacked later

    trained_model = federated_train(
        global_model=model,
        device_loaders=device_loaders,
        capable_devices=capable_devices,
        aggregation_method=aggregation_method,
        global_rounds=global_round,
        participation_rate=participation_rate, 
        local_epochs=local_epochs,
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        test_loader=test_loader
    )
    
    # NEW: Return data_dist as well
    return trained_model, test_loader, data_dist