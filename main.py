#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 15:20:14 2025

@author: zoz
"""

# In[1] ###################################### IMPORT LIBRARIES ######################################
import torch.nn as nn

import copy

from federated_learning_system import experiment_setting
from evaluation_functions import evaluate_model, calculate_sparsity, calculate_effective_size
from model_compression_methods import apply_pruning, apply_zero_aware_quantization



# In[2] ###################################### SETTINGS ######################################
# === User inputs ===
MODEL_NAME = 'VGG16'                # LeNet | AlexNet | VGG16
DATASET_NAME = 'CIFAR10'            # MNIST | FashionMNIST | CIFAR10
AGGREGATION_METHOD = 'avg'          # avg | avg_momentum | adagrad
GLOBAL_ROUND = 10                   # 100
TOTAL_DEVICES_IN_THE_SYSTEM = 100 
PARTICIPATION_RATE = 0.3            # (30% of randomly selected devices share their LM with the server per global round.)
PERCENTAGE_OF_CAPABLE_DEVICES = 0.2 # (Only 20% of devices can train GM.)
LOCAL_EPOCH = 1                     # 3, 5
BATCH_SIZE = 32                     # 64, 128 
OPTIMIZER = 'Adam'                  # 'SGD'
LEARNING_RATE = 0.01                # 0.001



print("""
###################################### EXPERIMENT SETTINGS ######################################
=== Benchmark Configuration ===
Model Name:               {}
Dataset:                  {}

=== Federated Learning Setup ===
Global Rounds:            {}
Total Devices:            {}
Participation Rate:       {}% ({} devices)
Capable Devices:          {}% ({} devices)

=== Training Configuration ===
Local Epochs:             {}
Batch Size:               {}
Optimizer:                {}
Learning Rate:            {}
Aggregation Method:       {}

#################################################################################################
""".format(
    MODEL_NAME, DATASET_NAME,
    GLOBAL_ROUND, TOTAL_DEVICES_IN_THE_SYSTEM, int(PARTICIPATION_RATE*100),
    int(TOTAL_DEVICES_IN_THE_SYSTEM * PARTICIPATION_RATE),
    int(PERCENTAGE_OF_CAPABLE_DEVICES*100),
    int(TOTAL_DEVICES_IN_THE_SYSTEM * PERCENTAGE_OF_CAPABLE_DEVICES),
    LOCAL_EPOCH, BATCH_SIZE, OPTIMIZER, LEARNING_RATE, AGGREGATION_METHOD
))



# In[3] ################################### FL TRAINING ###################################
# === Train model once ===
trained_model, test_loader = experiment_setting(
    model_name=MODEL_NAME,
    dataset_name=DATASET_NAME,
    aggregation_method=AGGREGATION_METHOD,
    num_rounds=GLOBAL_ROUND,
    total_num_of_devices=TOTAL_DEVICES_IN_THE_SYSTEM,
    participation_rate = PARTICIPATION_RATE,
    percentage_of_capable_devices=PERCENTAGE_OF_CAPABLE_DEVICES,
    local_epochs=LOCAL_EPOCH,
    batch_size=BATCH_SIZE,
    optimizer=OPTIMIZER,
    learning_rate=LEARNING_RATE
)

# ############################### EVALUATE TRAINED GM ################################
# Move model to CPU for sparsification/quantization
trained_model = trained_model.cpu()
# Take performance metrics of trained global model (trained_model)
tm_metrics = evaluate_model(trained_model, test_loader, nn.CrossEntropyLoss())
# Take sparsity of trained global model
tm_sparsity = calculate_sparsity(trained_model)
# Take the size of trained global model
tm_size = calculate_effective_size(trained_model) / (1024**2)  # MB


print("""
############################### EVALUATION RESULTS OF TRAINED GLOBEL MODEL ###############################
=== Model Performance ===
Loss:         {:.4f}
Accuracy (%): {:.2}
Precision (%):{:.2}
Recall (%):   {:.2}
F1 Score (%): {:.2}

=== Model Characteristics ===
Sparsity (%): {:.2}
Size:         {:.2f} MB
###############################################################################
""".format(
    tm_metrics['loss'],
    tm_metrics['accuracy'],
    tm_metrics['precision'],
    tm_metrics['recall'],
    tm_metrics['f1'],
    tm_sparsity,
    tm_size
))





# In[4] ############################### Model Compression: Trained-GM -> SM -> QM ################################
# For each sparsification amount, do:
for AMOUNT in range(1, 10, 1):
    AMOUNT = AMOUNT/10

    # Apply sparsification to original model
    sparsified_model = copy.deepcopy(trained_model)
    sparsified_model = apply_pruning(sparsified_model, amount=AMOUNT)
            
    # Quantize the sparsified model
    quantized_model = apply_zero_aware_quantization(sparsified_model)

    # Evaluate models
    sm_metrics = evaluate_model(sparsified_model, test_loader, nn.CrossEntropyLoss())
    sm_sparsity = calculate_sparsity(sparsified_model)
            
    qm_metrics = evaluate_model(quantized_model, test_loader, nn.CrossEntropyLoss())
    qm_sparsity = calculate_sparsity(quantized_model)

    # Calculate model sizes in MB
    tm_size = sum(p.numel() * 4 for p in trained_model.parameters())  # Original 32-bit model
    sm_size = sum(p.numel() * 4 for p in sparsified_model.parameters())  # Sparsified 32-bit model
    qm_size = calculate_effective_size(quantized_model)  # Quantized 8-bit model

    # Print results
    print(f"\n[SM->QM] Model Comparison when sparsification amount {AMOUNT}:")
    print(f"{'Metric':<14} | {'TM':<10} | {'SM':<10} | {'QM':<10}")
    print("-"*49)
    print(f"{'Loss':<14} | {tm_metrics['loss']:>10.4f} | {sm_metrics['loss']:>10.4f} | {qm_metrics['loss']:>10.4f}")
    print(f"{'Accuracy (%)':<14} | {tm_metrics['accuracy']:>10.2} | {sm_metrics['accuracy']:>10.2} | {qm_metrics['accuracy']:>10.2}")
    print(f"{'Precision (%)':<14} | {tm_metrics['precision']:>10.2} | {sm_metrics['precision']:>10.2} | {qm_metrics['precision']:>10.2}")
    print(f"{'Recall (%)':<14} | {tm_metrics['recall']:>10.2} | {sm_metrics['recall']:>10.2} | {qm_metrics['recall']:>10.2}")
    print(f"{'F1 Score (%)':<14} | {tm_metrics['f1']:>10.2} | {sm_metrics['f1']:>10.2} | {qm_metrics['f1']:>10.2}")
    print(f"{'Sparsity (%)':<14} | {tm_sparsity:>10.2f} | {sm_sparsity:>10.2f} | {qm_sparsity:>10.2f}")
    print(f"{'Size (MB)':<14} | {tm_size/(1024**2):>10.2f} | {sm_size/(1024**2):>10.2f} | {qm_size/(1024**2):>10.2f}")



print("\n########################################################################")
print("\n SM -> QM compression order is completed, now QM -> SM will start: \n")
print("########################################################################\n")


# In[5] ############################### Model Compression: Trained-GM -> QM -> SM ################################
# Apply quantization first (QM)
quantized_model = apply_zero_aware_quantization(trained_model)
qm_metrics = evaluate_model(quantized_model, test_loader, nn.CrossEntropyLoss())
qm_sparsity = calculate_sparsity(quantized_model)

# For each sparsification amount, do:
for AMOUNT in range(1, 10, 1):

    AMOUNT = AMOUNT/10

    # Apply structured sparsification using PyTorch pruning (SM)
    sparsified_model = copy.deepcopy(quantized_model)
    sparsified_model = apply_pruning(sparsified_model, amount=AMOUNT)
            
    sm_metrics = evaluate_model(sparsified_model, test_loader, nn.CrossEntropyLoss())
    sm_sparsity = calculate_sparsity(sparsified_model)
            
    # Calculate model sizes
    tm_size = sum(p.numel() * 4 for p in trained_model.parameters())  # Original 32-bit model
    qm_size = calculate_effective_size(quantized_model)               # Quantized 8-bit model
    sm_size = qm_size                                                 # Sparsified quantized model
            
    # Print results
    print(f"\n[QM->SM] Model Comparison when sparsification amount set to {AMOUNT}:")
    print(f"{'Metric':<12} | {'TM':<10} | {'QM':<10} | {'SM':<10}")
    print("-"*46)
    print(f"{'Loss':<12} | {tm_metrics['loss']:>10.4f} | {qm_metrics['loss']:>10.4f} | {sm_metrics['loss']:>10.4f}")
    print(f"{'Accuracy (%)':<12} | {tm_metrics['accuracy']:>10.4f} | {qm_metrics['accuracy']:>10.4f} | {sm_metrics['accuracy']:>10.4f}")
    print(f"{'Precision (%)':<12} | {tm_metrics['precision']:>10.4f} | {qm_metrics['precision']:>10.4f} | {sm_metrics['precision']:>10.4f}")
    print(f"{'Recall (%)':<12} | {tm_metrics['recall']:>10.4f} | {qm_metrics['recall']:>10.4f} | {sm_metrics['recall']:>10.4f}")
    print(f"{'F1 Score (%)':<12} | {tm_metrics['f1']:>10.4f} | {qm_metrics['f1']:>10.4f} | {sm_metrics['f1']:>10.4f}")
    print(f"{'Sparsity (%)':<12} | {tm_sparsity:>10.2f} | {qm_sparsity:>10.2f} | {sm_sparsity:>10.2f}")
    print(f"{'Size (MB)':<14} | {tm_size/(1024**2):>10.2f} | {sm_size/(1024**2):>10.2f} | {qm_size/(1024**2):>10.2f}")





