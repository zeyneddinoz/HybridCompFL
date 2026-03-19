#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:38:04 2025

@author: zoz
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch


# Dataset preparation 
def prepare_datasets(dataset_name: str, 
                    num_devices: int = 100,
                    batch_size: int = 32,
                    RESIZE_SIZE_FOR_VGG16: int = None,  # None for original size 32x32
                    alpha: float = 0.5):  
    
    
    if dataset_name in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        input_channels = 1
    elif dataset_name == 'CIFAR10':
        transforms_list = [
            transforms.RandomCrop(32, padding=4),  # Augmentation for better convergence
            transforms.RandomHorizontalFlip(),     # Augmentation
            transforms.ToTensor()
        ]
        if RESIZE_SIZE_FOR_VGG16:
            transforms_list.append(transforms.Resize(RESIZE_SIZE_FOR_VGG16))
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transforms_list)
        input_channels = 3
    else:
        raise ValueError("Unsupported dataset")

    dataset_map = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'CIFAR10': datasets.CIFAR10
    }

    # Load train and test separately
    train_dataset = dataset_map[dataset_name](root='./data', train=True, download=True, transform=transform)
    test_dataset = dataset_map[dataset_name](root='./data', train=False, download=True, transform=transform)

    # Dirichlet non-IID partitioning for train dataset
    labels = torch.tensor(train_dataset.targets)
    num_classes = len(torch.unique(labels))
    class_indices = [torch.where(labels == c)[0].tolist() for c in range(num_classes)]
    
    client_indices = [[] for _ in range(num_devices)]
    
    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        proportions = np.random.dirichlet([alpha] * num_devices)
        proportions = np.cumsum(proportions * len(class_indices[c])).astype(int)[:-1]
        splits = np.split(class_indices[c], proportions)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split)
    
    train_subsets = [Subset(train_dataset, indices) for indices in client_indices]

    # Return device loaders, test loader, input channels, client indices, and train dataset
    return (
        [DataLoader(s, batch_size=batch_size, shuffle=True) for s in train_subsets],
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        input_channels,
        client_indices,
        train_dataset  # New: return train_dataset for accessing targets
    )

# #################################


"""
# Dataset preparation 
def prepare_datasets_noverlap_but_equal_amount(dataset_name: str, 
                                               num_devices: int = 100,
                                               batch_size: int = 32):
    
    
    if dataset_name in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        input_channels = 1
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        input_channels = 3
    else:
        raise ValueError("Unsupported dataset")

    dataset_map = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'CIFAR10': datasets.CIFAR10
    }

    full_dataset = ConcatDataset([
        dataset_map[dataset_name](root='./data', train=True, download=True, transform=transform),
        dataset_map[dataset_name](root='./data', train=False, download=True, transform=transform)
    ])

    # Calculate subset sizes to match dataset length
    total_size = len(full_dataset)
    base_size = total_size // num_devices
    remainder = total_size % num_devices
    subset_sizes = [base_size + 1 if i < remainder else base_size for i in range(num_devices)]
    
    # Verify sum of sizes
    if sum(subset_sizes) != total_size:
        raise ValueError(f"Subset sizes sum ({sum(subset_sizes)}) does not match dataset size ({total_size})")
    
    subsets = random_split(full_dataset, subset_sizes)
    
    return (
        [DataLoader(s, batch_size=batch_size, shuffle=True) for s in subsets],
        DataLoader(ConcatDataset(subsets[10:]), batch_size=batch_size, shuffle=False),
        input_channels
    )
"""