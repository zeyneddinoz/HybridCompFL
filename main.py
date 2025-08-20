#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 15:20:14 2025

@author: zoz
"""

# In[1] ###################################### IMPORT LIBRARIES ######################################
import torch.nn as nn
import torch
import os


from federated_learning_system import experiment_setting
from evaluation_functions import evaluate_model, calculate_sparsity  # No effective_size
from model_compression_methods import apply_pruning, apply_zero_aware_quantization, make_sparse, make_dequant


# In[2] ###################################### SETTINGS ######################################
# === User inputs ===
MODEL_NAME = 'VGG16'                # LeNet | AlexNet | VGG16
DATASET_NAME = 'CIFAR10'            # MNIST | FashionMNIST | CIFAR10
AGGREGATION_METHOD = 'avg'          # avg | avg_momentum | adagrad
GLOBAL_ROUND = 10                   # 100
TOTAL_DEVICES_IN_THE_SYSTEM = 100   # 50
PARTICIPATION_RATE = 0.3            # (30% of randomly selected devices share their LM with the server per global round.)
PERCENTAGE_OF_CAPABLE_DEVICES = 0.2 # (Only 20% of devices can train GM.)
LOCAL_EPOCH = 1                     # 3, 5
BATCH_SIZE = 32                     # 64, 128 
OPTIMIZER = 'Adam'                  # 'SGD'
LEARNING_RATE = 0.01                # 0.001

# Initialize results dictionary
results = {
    "experiment_setting": {
        "model_configuration": {
            "model_name": MODEL_NAME,
            "dataset_name": DATASET_NAME
        },
        "fl_setup": {
            "global_rounds": GLOBAL_ROUND,
            "total_devices": TOTAL_DEVICES_IN_THE_SYSTEM,
            "participation_rate": PARTICIPATION_RATE,
            "num_participating_devices": int(TOTAL_DEVICES_IN_THE_SYSTEM * PARTICIPATION_RATE),
            "percentage_of_capable_devices": PERCENTAGE_OF_CAPABLE_DEVICES,
            "num_capable_devices": int(TOTAL_DEVICES_IN_THE_SYSTEM * PERCENTAGE_OF_CAPABLE_DEVICES)
        },
        "training_configuration": {
            "local_epoch": LOCAL_EPOCH,
            "batch_size": BATCH_SIZE,
            "optimizer": OPTIMIZER,
            "learning_rate": LEARNING_RATE,
            "aggregation_method": AGGREGATION_METHOD
        }
    },
    "trained_global_model": {},
    "compression_SM_QM": [],
    "compression_QM_SM": [],
    "saved_models": {}  # Add a key to store model paths
}

# Print settings from results dictionary
setting = results["experiment_setting"]
model_conf = setting["model_configuration"]
fl_setup = setting["fl_setup"]
train_conf = setting["training_configuration"]

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
    model_conf["model_name"], model_conf["dataset_name"],
    fl_setup["global_rounds"], fl_setup["total_devices"], 
    int(fl_setup["participation_rate"]*100), fl_setup["num_participating_devices"],
    int(fl_setup["percentage_of_capable_devices"]*100), fl_setup["num_capable_devices"],
    train_conf["local_epoch"], train_conf["batch_size"], 
    train_conf["optimizer"], train_conf["learning_rate"], 
    train_conf["aggregation_method"]
))

# In[3] ################################### FL TRAINING ###################################
# === Train model once ===
trained_model, test_loader = experiment_setting(
    model_name=MODEL_NAME,
    dataset_name=DATASET_NAME,
    aggregation_method=AGGREGATION_METHOD,
    global_round=GLOBAL_ROUND,
    total_num_of_devices=TOTAL_DEVICES_IN_THE_SYSTEM,
    participation_rate = PARTICIPATION_RATE,
    percentage_of_capable_devices=PERCENTAGE_OF_CAPABLE_DEVICES,
    local_epochs=LOCAL_EPOCH,
    batch_size=BATCH_SIZE,
    optimizer=OPTIMIZER,
    learning_rate=LEARNING_RATE
)

# Save and evaluate TM with file size
tm_filename = "trained_global_model.pth"
torch.save(trained_model.state_dict(), tm_filename)
tm_size = os.path.getsize(tm_filename) / (1024**2)  # MB
tm_metrics = evaluate_model(trained_model, test_loader, nn.CrossEntropyLoss())
tm_sparsity = calculate_sparsity(trained_model)

results["trained_global_model"] = {
    "loss": tm_metrics['loss'],
    "accuracy": round(tm_metrics['accuracy'] * 100, 2),
    "precision": round(tm_metrics['precision'] * 100, 2),
    "recall": round(tm_metrics['recall'] * 100, 2),
    "f1": round(tm_metrics['f1'] * 100, 2),
    "sparsity": round(tm_sparsity, 2),
    "size": round(tm_size, 2)
}


# Print from results dictionary
tm = results["trained_global_model"]
print("""
############################### EVALUATION RESULTS OF TRAINED GLOBEL MODEL ###############################
=== Model Performance ===
Loss:         {:.4f}
Accuracy (%): {:.2f}%
Precision (%):{:.2f}%
Recall (%):   {:.2f}%
F1 Score (%): {:.2f}%

=== Model Characteristics ===
Sparsity (%): {:.2f}%
Size:         {:.2f} MB
###############################################################################
""".format(
    tm['loss'],
    tm['accuracy'],
    tm['precision'],
    tm['recall'],
    tm['f1'],
    tm['sparsity'],
    tm['size']
))

# In[4] ############################### Model Compression: Trained-GM -> SM -> QM ################################
for AMOUNT in range(1, 10, 1):
    AMOUNT_val = AMOUNT/10

    # Apply pruning (dense with zeros)
    sparsified_model = apply_pruning(trained_model, amount=AMOUNT_val)
    
    # Save sparse version for compression
    sparse_sm = make_sparse(sparsified_model)
    sm_filename = f"sm_then_qm_sparsed_model_{AMOUNT_val}.pth"
    torch.save(sparse_sm.state_dict(), sm_filename)
    sm_size = os.path.getsize(sm_filename) / (1024**2)
            
    # Quantize the dense pruned model
    quantized_model = apply_zero_aware_quantization(sparsified_model)
    
    # Save quantized
    qm_filename = f"sm_then_qm_quantized_model_{AMOUNT_val}.pth"
    torch.save(quantized_model.state_dict(), qm_filename)
    qm_size = os.path.getsize(qm_filename) / (1024**2)
    
    # Store model paths in results
    if f"amount_{AMOUNT_val}" not in results["saved_models"]:
        results["saved_models"][f"amount_{AMOUNT_val}"] = {}
    results["saved_models"][f"amount_{AMOUNT_val}"]["sm_then_qm_sparsed"] = sm_filename
    results["saved_models"][f"amount_{AMOUNT_val}"]["sm_then_qm_quantized"] = qm_filename

    # Evaluate (use dense/dequant versions)
    sm_metrics = evaluate_model(sparsified_model, test_loader, nn.CrossEntropyLoss())
    sm_sparsity = calculate_sparsity(sparsified_model)    
    qm_metrics = evaluate_model(quantized_model, test_loader, nn.CrossEntropyLoss())
    qm_sparsity = calculate_sparsity(quantized_model)
    
    
    # TM size (unchanged, but now file-based)
    tm_size = os.path.getsize("trained_global_model.pth") / (1024**2)
    
    
    # Store results
    compression_result = {
        "amount": round(AMOUNT_val, 2),
        "TM": {
            "loss": tm_metrics['loss'],
            "accuracy": round(tm_metrics['accuracy'] * 100, 2),
            "precision": round(tm_metrics['precision'] * 100, 2),
            "recall": round(tm_metrics['recall'] * 100, 2),
            "f1": round(tm_metrics['f1'] * 100, 2),
            "sparsity": round(tm_sparsity, 2),
            "size": round(tm_size, 2)
        },
        "SM": {
            "loss": sm_metrics['loss'],
            "accuracy": round(sm_metrics['accuracy'] * 100, 2),
            "precision": round(sm_metrics['precision'] * 100, 2),
            "recall": round(sm_metrics['recall'] * 100, 2),
            "f1": round(sm_metrics['f1'] * 100, 2),
            "sparsity": round(sm_sparsity, 2),
            "size": round(sm_size, 2)
        },
        "QM": {
            "loss": qm_metrics['loss'],
            "accuracy": round(qm_metrics['accuracy'] * 100, 2),
            "precision": round(qm_metrics['precision'] * 100, 2),
            "recall": round(qm_metrics['recall'] * 100, 2),
            "f1": round(qm_metrics['f1'] * 100, 2),
            "sparsity": round(qm_sparsity, 2),
            "size": round(qm_size, 2)
        }
    }
    results["compression_SM_QM"].append(compression_result)
    
    # Print from stored results
    cr = compression_result
    print(f"\n[SM->QM] Model Comparison when sparsification amount {cr['amount']:.1f}:")
    print(f"{'Metric':<14} | {'TM':<10} | {'SM':<10} | {'QM':<10}")
    print("-"*49)
    print(f"{'Loss':<14} | {cr['TM']['loss']:>10.4f} | {cr['SM']['loss']:>10.4f} | {cr['QM']['loss']:>10.4f}")
    print(f"{'Accuracy (%)':<14} | {cr['TM']['accuracy']:>10.2f} | {cr['SM']['accuracy']:>10.2f} | {cr['QM']['accuracy']:>10.2f}")
    print(f"{'Precision (%)':<14} | {cr['TM']['precision']:>10.2f} | {cr['SM']['precision']:>10.2f} | {cr['QM']['precision']:>10.2f}")
    print(f"{'Recall (%)':<14} | {cr['TM']['recall']:>10.2f} | {cr['SM']['recall']:>10.2f} | {cr['QM']['recall']:>10.2f}")
    print(f"{'F1 Score (%)':<14} | {cr['TM']['f1']:>10.2f} | {cr['SM']['f1']:>10.2f} | {cr['QM']['f1']:>10.2f}")
    print(f"{'Sparsity (%)':<14} | {cr['TM']['sparsity']:>10.2f} | {cr['SM']['sparsity']:>10.2f} | {cr['QM']['sparsity']:>10.2f}")
    print(f"{'Size (MB)':<14} | {cr['TM']['size']:>10.2f} | {cr['SM']['size']:>10.2f} | {cr['QM']['size']:>10.2f}")

print("\n########################################################################")
print("\n SM -> QM compression order is completed, now QM -> SM will start: \n")
print("########################################################################\n")

# In[5] ############################### Model Compression: Trained-GM -> QM -> SM ################################
# Quantize first
quantized_model = apply_zero_aware_quantization(trained_model)

# Save base QM
base_qm_filename = "qm_then_sm_base_quantized_model.pth"
torch.save(quantized_model.state_dict(), base_qm_filename)
qm_size = os.path.getsize(base_qm_filename) / (1024**2)

qm_metrics = evaluate_model(quantized_model, test_loader, nn.CrossEntropyLoss())
qm_sparsity = calculate_sparsity(quantized_model)

# Store QM base results
base_qm = {
    "loss": qm_metrics['loss'],
    "accuracy": round(qm_metrics['accuracy'] * 100, 2),
    "precision": round(qm_metrics['precision'] * 100, 2),
    "recall": round(qm_metrics['recall'] * 100, 2),
    "f1": round(qm_metrics['f1'] * 100, 2),
    "sparsity": round(qm_sparsity, 2),
    "size": round(qm_size, 2)
}

# For each sparsification amount, do:
for AMOUNT in range(1, 10, 1):
    AMOUNT_val = AMOUNT/10  
    
    # Dequant for pruning
    dequant_model = make_dequant(quantized_model)
    
    # Apply pruning on dequant
    sparsified_model = apply_pruning(dequant_model, amount=AMOUNT_val)
    
    # Make sparse for save
    sparse_sm = make_sparse(sparsified_model)
    sm_filename = f"qm_then_sm_sparsed_model_{AMOUNT_val}.pth"
    torch.save(sparse_sm.state_dict(), sm_filename)
    sm_size = os.path.getsize(sm_filename) / (1024**2)
   
    # Store model path in results
    if f"amount_{AMOUNT_val}" not in results["saved_models"]:
        results["saved_models"][f"amount_{AMOUNT_val}"] = {}
    results["saved_models"][f"amount_{AMOUNT_val}"]["qm_then_sm_sparsed"] = sm_filename
            
    sm_metrics = evaluate_model(sparsified_model, test_loader, nn.CrossEntropyLoss())
    sm_sparsity = calculate_sparsity(sparsified_model)

    # TM size file-based
    tm_size = os.path.getsize("trained_global_model.pth") / (1024**2)

    # Store results
    compression_result = {
        "amount": round(AMOUNT_val, 2),
        "TM": {
            "loss": tm_metrics['loss'],
            "accuracy": round(tm_metrics['accuracy'] * 100, 2),
            "precision": round(tm_metrics['precision'] * 100, 2),
            "recall": round(tm_metrics['recall'] * 100, 2),
            "f1": round(tm_metrics['f1'] * 100, 2),
            "sparsity": round(tm_sparsity, 2),
            "size": round(tm_size, 2)
        },
        "QM": base_qm,
        "SM": {
            "loss": sm_metrics['loss'],
            "accuracy": round(sm_metrics['accuracy'] * 100, 2),
            "precision": round(sm_metrics['precision'] * 100, 2),
            "recall": round(sm_metrics['recall'] * 100, 2),
            "f1": round(sm_metrics['f1'] * 100, 2),
            "sparsity": round(sm_sparsity, 2),
            "size": round(sm_size, 2)
        }
    }
    results["compression_QM_SM"].append(compression_result)
    
    # Print from stored results
    cr = compression_result
    print(f"\n[QM->SM] Model Comparison when sparsification amount set to {cr['amount']:.1f}:")
    print(f"{'Metric':<14} | {'TM':<10} | {'QM':<10} | {'SM':<10}")
    print("-"*52)
    print(f"{'Loss':<14} | {cr['TM']['loss']:>10.4f} | {cr['QM']['loss']:>10.4f} | {cr['SM']['loss']:>10.4f}")
    print(f"{'Accuracy (%)':<14} | {cr['TM']['accuracy']:>10.2f} | {cr['QM']['accuracy']:>10.2f} | {cr['SM']['accuracy']:>10.2f}")
    print(f"{'Precision (%)':<14} | {cr['TM']['precision']:>10.2f} | {cr['QM']['precision']:>10.2f} | {cr['SM']['precision']:>10.2f}")
    print(f"{'Recall (%)':<14} | {cr['TM']['recall']:>10.2f} | {cr['QM']['recall']:>10.2f} | {cr['SM']['recall']:>10.2f}")
    print(f"{'F1 Score (%)':<14} | {cr['TM']['f1']:>10.2f} | {cr['QM']['f1']:>10.2f} | {cr['SM']['f1']:>10.2f}")
    print(f"{'Sparsity (%)':<14} | {cr['TM']['sparsity']:>10.2f} | {cr['QM']['sparsity']:>10.2f} | {cr['SM']['sparsity']:>10.2f}")
    print(f"{'Size (MB)':<14} | {cr['TM']['size']:>10.2f} | {cr['QM']['size']:>10.2f} | {cr['SM']['size']:>10.2f}")


# In[6] DONE - VISUALIZATION FUNCTION WILL BE ADDED LATER.


