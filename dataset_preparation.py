#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:38:04 2025

@author: zoz
"""


from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split



# Dataset preparation 
def prepare_datasets(dataset_name: str, num_devices: int = 100):
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

    subset_size = len(full_dataset) // num_devices
    subsets = random_split(full_dataset, [subset_size] * num_devices)
    
    return (
        [DataLoader(s, batch_size=32, shuffle=True) for s in subsets],
        DataLoader(ConcatDataset(subsets[10:]), batch_size=32, shuffle=False),
        input_channels
    )