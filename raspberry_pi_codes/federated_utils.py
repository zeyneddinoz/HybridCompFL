#Importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
import numpy as np
import random
from typing import Callable, List, Dict
from collections import defaultdict
import copy
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import datetime
import flwr as fl


# 1- settings:
###################################### SETTINGS ######################################
# === User inputs ===
MODEL_NAME = 'LeNet'                # LeNet | AlexNet | VGG16
DATASET_NAME = 'MNIST'            # MNIST | FashionMNIST | CIFAR10
AGGREGATION_METHOD = 'avg'          # avg | avg_momentum | adagrad
GLOBAL_ROUND = 5                   # 100
TOTAL_DEVICES_IN_THE_SYSTEM = 50   # 100, 50
PARTICIPATION_RATE = 0.6            # (60% of randomly selected devices share their LM with the server per global round.)
PERCENTAGE_OF_CAPABLE_DEVICES = 0.1 # (Only 10% of devices can train GM.)
LOCAL_EPOCH = 5                     # 3, 5
BATCH_SIZE = 64                     # 64, 128 
OPTIMIZER = 'Adam'                  # 'SGD'
LEARNING_RATE = 0.001                # 0.001
ALPHA = 0.5
RESIZE_SIZE_FOR_VGG16=224
MODE = 'simulation'                 # 'simulation' or 'server'





# 2- Dataset Preparetion Function:
def analyze_distribution( dataset_name, 
        total_num_of_devices, 
        batch_size, 
        RESIZE_SIZE_FOR_VGG16,
        alpha,percentage_of_capable_devices):
    seed=42
    # Ensure reproducibility across all devices
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # Receive train_dataset along with other returns
    device_loaders, test_loader, input_channels, client_indices, train_dataset = prepare_datasets(
        dataset_name, 
        total_num_of_devices, 
        batch_size, 
        RESIZE_SIZE_FOR_VGG16,
        alpha
    )
  
    # model = get_model(model_name, input_channels=input_channels)
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
    # print(f'data_dist: {data_dist}')
    # NEW: Return data_dist as well
    return data_dist


# Dataset preparation 
def prepare_datasets(dataset_name: str, 
                    num_devices: int = 100,
                    batch_size: int = 32,
                    RESIZE_SIZE_FOR_VGG16: int = None,  # None for original size 32x32
                    alpha: float = 0.5):  
    seed=42
    # Ensure reproducibility across all devices
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
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
# 3- Model Aggregation Strategies:

# Aggregation strategies 
class Aggregator:
    def __init__(self):
        self.server_momentum = None
        self.server_opt_state = defaultdict(float)
        
    def fed_avg(self, local_updates: List[dict]) -> dict:
        avg_weights = {}
        for key in local_updates[0].keys():
            avg_weights[key] = torch.stack(
                [update[key].float() for update in local_updates]
            ).mean(dim=0)
        return avg_weights

    def fed_avg_momentum(self, local_updates: List[dict], beta=0.9) -> dict:
        current_update = self.fed_avg(local_updates)
        if self.server_momentum is None:
            self.server_momentum = current_update
        else:
            for key in current_update:
                self.server_momentum[key] = beta * self.server_momentum[key] + (1 - beta) * current_update[key]
        return self.server_momentum

    def fed_adagrad(self, local_updates: List[dict], eta=0.01, epsilon=1e-8) -> dict:
        current_update = self.fed_avg(local_updates)
        for key in current_update:
            self.server_opt_state[key] += current_update[key].float() ** 2
            current_update[key] = eta * current_update[key] / (torch.sqrt(self.server_opt_state[key]) + epsilon)
        return current_update
# 4- Model Architectures:

# Model definitions (from first code)
def get_model(model_name: str, input_channels: int = 1, num_classes: int = 10) -> nn.Module:
    class LeNet(nn.Module):
        def __init__(self, input_channels=input_channels):
            super(LeNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Added adaptive pooling
            self.classifier = nn.Sequential(
                nn.Linear(16*4*4, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.adaptive_pool(x)  # Adaptive pooling to fixed size
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    class AlexNet(nn.Module):
        def __init__(self, input_channels=1):
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))  # Changed to adaptive pooling
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.adaptive_pool(x)  # Adaptive pooling to fixed size
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    class VGG16(nn.Module):
        def __init__(self, input_channels=input_channels):
            super(VGG16, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Added adaptive pooling
            self.classifier = nn.Sequential(
                nn.Linear(256*4*4, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.adaptive_pool(x)  # Adaptive pooling to fixed size
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    models = {
        'LeNet': LeNet(input_channels=input_channels),
        'AlexNet': AlexNet(input_channels=input_channels),
        'VGG16': VGG16(input_channels=input_channels)
    }

    return models[model_name].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# 5- Federated Learning Training:

# # Federated training loop (from first code)
# def federated_train(
#     global_model: nn.Module,
#     device_loaders: List[DataLoader],
#     selected_devices: List[int],
#     aggregation_method: str,
#     num_rounds: int = 10,
#     participation_rate: float = 0.3,
#     local_epochs: int = 3,
#     test_loader: DataLoader = None
# ) -> nn.Module:
    
#     device = next(global_model.parameters()).device
#     num_selected = int(len(selected_devices) * participation_rate)
#     aggregator = Aggregator()
#     aggregation_fn = getattr(aggregator, f'fed_{aggregation_method.lower()}')

#     for round_num in range(num_rounds):
#         print(f"\nGlobal Round {round_num+1}/{num_rounds}")
        
#         # Select devices and collect updates
#         selected = random.sample(selected_devices, num_selected)
#         local_updates = []
        
#         for dev_idx in selected:
#             local_model = type(global_model)()
#             local_model.load_state_dict(global_model.state_dict())
#             local_model.to(device)
            
#             optimizer = optim.Adam(local_model.parameters())
#             local_model.train()
            
#             for _ in range(local_epochs):
#                 for X, y in device_loaders[dev_idx]:
#                     X, y = X.to(device), y.to(device)
#                     optimizer.zero_grad()
#                     loss = nn.CrossEntropyLoss()(local_model(X), y)
#                     loss.backward()
#                     optimizer.step()
            
#             local_updates.append(local_model.state_dict())
        
#         # Aggregate updates
#         global_weights = aggregation_fn(local_updates)
#         global_model.load_state_dict(global_weights)
        
#         # Evaluation
#         if test_loader:
#             global_model.eval()
#             correct, total = 0, 0
#             with torch.no_grad():
#                 for X, y in test_loader:
#                     X, y = X.to(device), y.to(device)
#                     correct += (global_model(X).argmax(1) == y).sum().item()
#                     total += y.size(0)
#             print(f"Test Accuracy: {100*correct/total:.2f}%")
    
#     return global_model

# def experiment_setting(
#     model_name: str = 'AlexNet',
#     dataset_name: str = 'MNIST',
#     aggregation_method: str = 'avg',
#     num_rounds: int = 10,
#     participation_rate = 1.0,
#     local_epochs: int = 3,
# ):
#     device_loaders, test_loader, input_channels = prepare_datasets(dataset_name)
#     model = get_model(model_name, input_channels=input_channels)
    
#     # Determine capable device count based on dataset
#     if dataset_name == 'CIFAR10':
#         num_capable_devices = 20  # 20% of 100 devices
#     else:  # MNIST or FashionMNIST
#         num_capable_devices = 10  # 10% of 100 devices
    
#     group1_devices = list(range(num_capable_devices))  # Capable devices
    
#     trained_model = federated_train(
#         global_model=model,
#         device_loaders=device_loaders,
#         selected_devices=group1_devices,
#         aggregation_method=aggregation_method,
#         num_rounds=num_rounds,
#         participation_rate=participation_rate, 
#         test_loader=test_loader,
#         local_epochs = 3
#     )
    
#     return trained_model, test_loader



# 6- Compression Functions:


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


# 7- Evaluation Functions:

### 7.1- Sparsity Percentage:

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

### 7.3 Measure Loss, Accuracy, Precision, Recall, and F1 Score
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
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    }

# Note: calculate_effective_size is removed; use actual file size instead