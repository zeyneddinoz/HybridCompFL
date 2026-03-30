import flwr as fl
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime  # Added for timestamp
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from federated_utils import (
    get_model, prepare_datasets, Aggregator,
    evaluate_model,apply_zero_aware_quantization, apply_pruning,
    calculate_sparsity, make_sparse, make_dequant, analyze_distribution
)

from federated_utils import (
    MODEL_NAME, DATASET_NAME, AGGREGATION_METHOD, GLOBAL_ROUND,
    TOTAL_DEVICES_IN_THE_SYSTEM, PARTICIPATION_RATE,
    PERCENTAGE_OF_CAPABLE_DEVICES, LOCAL_EPOCH, BATCH_SIZE,
    OPTIMIZER, LEARNING_RATE, ALPHA,RESIZE_SIZE_FOR_VGG16)

# NEW: Create experiment folder
experiment_folder = f"{MODEL_NAME}_{DATASET_NAME}"
os.makedirs(experiment_folder, exist_ok=True)

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
            "num_participating_devices": int(TOTAL_DEVICES_IN_THE_SYSTEM * PERCENTAGE_OF_CAPABLE_DEVICES * PARTICIPATION_RATE),
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

initial_config = {
     "model_name": MODEL_NAME,
     "dataset_name": DATASET_NAME
 }
# --- Prepare data and global model ---
DEVICE_LOADERS, TEST_LOADER, INPUT_CHANNELS,_,_ = prepare_datasets(DATASET_NAME,TOTAL_DEVICES_IN_THE_SYSTEM, BATCH_SIZE, RESIZE_SIZE_FOR_VGG16, ALPHA)
global_model = get_model(MODEL_NAME, input_channels=INPUT_CHANNELS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model.to(device)

global_weights = global_model.state_dict()
aggregator = Aggregator()
aggregation_fn = getattr(aggregator, f"fed_{AGGREGATION_METHOD}")

# --- Custom Strategy ---
class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.dataset_name = DATASET_NAME

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        config = {
            "server_round": server_round,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
        }

        fit_ins = fl.common.FitIns(parameters, config)
        num_available = len(client_manager.all())
        num_clients = max(int(self.fraction_fit * num_available), self.min_fit_clients)

        clients = client_manager.sample(
            num_clients=num_clients,
            min_num_clients=self.min_fit_clients
        )
        print(f"[SERVER] Round {server_round} - sending fit instructions to {len(clients)} clients")
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException]
    ) -> Tuple[fl.common.Parameters, Dict[str, fl.common.Scalar], Dict[str, fl.common.Scalar]]:
        global global_weights

        if failures:
            print(f"[SERVER] Round {server_round}: failures detected - {len(failures)}")

        updates = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Debug: Print weight checksum for each client
        for idx, update in enumerate(updates):
            checksum = sum([np.sum(w) for w in update])
            print(f"[SERVER][DEBUG] Round {server_round} - Client {idx} weights checksum: {checksum:.6f}")

        local_updates = []
        for update in updates:
            params_dict = zip(global_weights.keys(), update)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            local_updates.append(state_dict)

        global_weights = aggregation_fn(local_updates)
        global_model.load_state_dict(global_weights)

        # Evaluate
        metrics = evaluate_model(global_model, TEST_LOADER, nn.CrossEntropyLoss(), device=device)
        print(f"[SERVER] Round {server_round} Eval -> Acc: {metrics['accuracy']*100:.2f}% | Loss: {metrics['loss']:.4f}")

        return ndarrays_to_parameters([v.cpu().numpy() for v in global_weights.values()]), {}

    def aggregate_metrics(self, metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        if not metrics:
            return {}
        return {k: float(np.mean([m[k] for _, m in metrics])) for k in metrics[0][1].keys()}

# --- Evaluation Function ---
def evaluate_fn(server_round, parameters, config):
    model = get_model(MODEL_NAME, input_channels=INPUT_CHANNELS)
    
    # Safe handling: check type
    if isinstance(parameters, list):
        weights = parameters
    else:
        weights = parameters_to_ndarrays(parameters)

    state_dict = OrderedDict({
        k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)
    })
    model.load_state_dict(state_dict)
    model.to(device)

    metrics = evaluate_model(model, TEST_LOADER, nn.CrossEntropyLoss(), device=device)
    return float(metrics["loss"]), metrics


# --- Start Server ---
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=GLOBAL_ROUND),
    strategy=CustomStrategy(
        model_name=MODEL_NAME,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        evaluate_fn=evaluate_fn,
        min_evaluate_clients = 0,
        fraction_evaluate = 0.0,
    ),
)


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



print("\n[SERVER] Training complete. Running post-training pipeline...\n")

trained_model = global_model.cpu()

# NEW: Add to results
# results["data_distribution"] = data_dist

results["data_distribution"] = analyze_distribution( dataset_name=DATASET_NAME, 
        total_num_of_devices=TOTAL_DEVICES_IN_THE_SYSTEM, 
        batch_size=BATCH_SIZE, RESIZE_SIZE_FOR_VGG16=RESIZE_SIZE_FOR_VGG16,
        alpha=ALPHA,percentage_of_capable_devices=PERCENTAGE_OF_CAPABLE_DEVICES)

# Save and evaluate TM with file size
tm_filename = os.path.join(experiment_folder, "trained_global_model.pth")
torch.save(trained_model.state_dict(), tm_filename)
tm_size = os.path.getsize(tm_filename) / (1024**2)  # MB
tm_metrics = evaluate_model(trained_model, TEST_LOADER, nn.CrossEntropyLoss())
tm_sparsity = calculate_sparsity(trained_model)

results["trained_global_model"] = {
    "loss": tm_metrics['loss'],
    "accuracy": round(tm_metrics['accuracy'] * 100, 2),
    "precision": round(tm_metrics['precision'] * 100, 2),
    "recall": round(tm_metrics['recall'] * 100, 2),
    "f1": round(tm_metrics['f1'] * 100, 2),
    "f1_weighted": round(tm_metrics['f1_weighted'] * 100, 2),
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
F1 Weighted (%): {:.2f}%

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
    tm['f1_weighted'],
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
    sm_filename = os.path.join(experiment_folder, f"sm_then_qm_sparsed_model_{AMOUNT_val}.pth")
    torch.save(sparse_sm.state_dict(), sm_filename)
    sm_size = os.path.getsize(sm_filename) / (1024**2)
    
    # Store model path in results
    if f"amount_{AMOUNT_val}" not in results["saved_models"]:
        results["saved_models"][f"amount_{AMOUNT_val}"] = {}
    results["saved_models"][f"amount_{AMOUNT_val}"]["sm_then_qm_sparsed"] = sm_filename
    
    sm_metrics = evaluate_model(sparsified_model, TEST_LOADER, nn.CrossEntropyLoss())
    sm_sparsity = calculate_sparsity(sparsified_model)
    
    # Quantize the sparsified model
    quantized_model = apply_zero_aware_quantization(sparsified_model)
    
    # Save QM
    qm_filename = os.path.join(experiment_folder, f"sm_then_qm_quantized_model_{AMOUNT_val}.pth")
    torch.save(quantized_model.state_dict(), qm_filename)
    qm_size = os.path.getsize(qm_filename) / (1024**2)
    
    # Store model path
    results["saved_models"][f"amount_{AMOUNT_val}"]["sm_then_qm_quantized"] = qm_filename
    
    qm_metrics = evaluate_model(quantized_model, TEST_LOADER, nn.CrossEntropyLoss())
    qm_sparsity = calculate_sparsity(quantized_model)

    # Store results
    compression_result = {
        "amount": round(AMOUNT_val, 2),
        "TM": {
            "loss": tm_metrics['loss'],
            "accuracy": round(tm_metrics['accuracy'] * 100, 2),
            "precision": round(tm_metrics['precision'] * 100, 2),
            "recall": round(tm_metrics['recall'] * 100, 2),
            "f1": round(tm_metrics['f1'] * 100, 2),
            "f1_weighted": round(tm_metrics['f1_weighted'] * 100, 2),
            "sparsity": round(tm_sparsity, 2),
            "size": round(tm_size, 2)
        },
        "SM": {
            "loss": sm_metrics['loss'],
            "accuracy": round(sm_metrics['accuracy'] * 100, 2),
            "precision": round(sm_metrics['precision'] * 100, 2),
            "recall": round(sm_metrics['recall'] * 100, 2),
            "f1": round(sm_metrics['f1'] * 100, 2),
            "f1_weighted": round(sm_metrics['f1_weighted'] * 100, 2),
            "sparsity": round(sm_sparsity, 2),
            "size": round(sm_size, 2)
        },
        "QM": {
            "loss": qm_metrics['loss'],
            "accuracy": round(qm_metrics['accuracy'] * 100, 2),
            "precision": round(qm_metrics['precision'] * 100, 2),
            "recall": round(qm_metrics['recall'] * 100, 2),
            "f1": round(qm_metrics['f1'] * 100, 2),
            "f1_weighted": round(qm_metrics['f1_weighted'] * 100, 2),
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
    print(f"{'F1 Weighted (%)':<14} | {cr['TM']['f1_weighted']:>10.2f} | {cr['SM']['f1_weighted']:>10.2f} | {cr['QM']['f1_weighted']:>10.2f}")
    print(f"{'Sparsity (%)':<14} | {cr['TM']['sparsity']:>10.2f} | {cr['SM']['sparsity']:>10.2f} | {cr['QM']['sparsity']:>10.2f}")
    print(f"{'Size (MB)':<14} | {cr['TM']['size']:>10.2f} | {cr['SM']['size']:>10.2f} | {cr['QM']['size']:>10.2f}")

# NEW: Create and print summary table for SM->QM
data = {
    "Metric": ["Size (MB)", "Sparsity (%)", "Weighted F1 (%)"]
}
columns = ["TM"] + [f"SM{i}" for i in range(1, 10)]

# TM values
tm_size = results["trained_global_model"]["size"]
tm_sparsity = 0.0
tm_f1w = results["trained_global_model"]["f1_weighted"]
data["TM"] = [tm_size, tm_sparsity, tm_f1w]

# For each SM (QM in this path)
for i, entry in enumerate(results["compression_SM_QM"]):
    sm_num = i + 1
    qm = entry["QM"]
    size = qm["size"]
    sparsity = entry["amount"] * 100
    f1w = qm["f1_weighted"]
    data[f"SM{sm_num}"] = [size, sparsity, f1w]

df = pd.DataFrame(data)
df.set_index("Metric", inplace=True)

print("\nSummary Table for SM->QM:")
print(df.to_string())

# Save to results
results["sm_qm_summary_table"] = df.to_dict(orient='index')

print("\n########################################################################")
print("\n SM -> QM compression order is completed, now QM -> SM will start: \n")
print("########################################################################\n")

# In[5] ############################### Model Compression: Trained-GM -> QM -> SM ################################
# Quantize first
quantized_model = apply_zero_aware_quantization(trained_model)

# Save base QM
base_qm_filename = os.path.join(experiment_folder, "qm_then_sm_base_quantized_model.pth")
torch.save(quantized_model.state_dict(), base_qm_filename)
qm_size = os.path.getsize(base_qm_filename) / (1024**2)

qm_metrics = evaluate_model(quantized_model, TEST_LOADER, nn.CrossEntropyLoss())
qm_sparsity = calculate_sparsity(quantized_model)

# Store QM base results
base_qm = {
    "loss": qm_metrics['loss'],
    "accuracy": round(qm_metrics['accuracy'] * 100, 2),
    "precision": round(qm_metrics['precision'] * 100, 2),
    "recall": round(qm_metrics['recall'] * 100, 2),
    "f1": round(qm_metrics['f1'] * 100, 2),
    "f1_weighted": round(qm_metrics['f1_weighted'] * 100, 2),
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
    sm_filename = os.path.join(experiment_folder, f"qm_then_sm_sparsed_model_{AMOUNT_val}.pth")
    torch.save(sparse_sm.state_dict(), sm_filename)
    sm_size = os.path.getsize(sm_filename) / (1024**2)
   
    # Store model path in results
    if f"amount_{AMOUNT_val}" not in results["saved_models"]:
        results["saved_models"][f"amount_{AMOUNT_val}"] = {}
    results["saved_models"][f"amount_{AMOUNT_val}"]["qm_then_sm_sparsed"] = sm_filename
            
    sm_metrics = evaluate_model(sparsified_model, TEST_LOADER, nn.CrossEntropyLoss())
    sm_sparsity = calculate_sparsity(sparsified_model)

    # TM size file-based
    tm_size = os.path.getsize(os.path.join(experiment_folder, "trained_global_model.pth")) / (1024**2)

    # Store results
    compression_result = {
        "amount": round(AMOUNT_val, 2),
        "TM": {
            "loss": tm_metrics['loss'],
            "accuracy": round(tm_metrics['accuracy'] * 100, 2),
            "precision": round(tm_metrics['precision'] * 100, 2),
            "recall": round(tm_metrics['recall'] * 100, 2),
            "f1": round(tm_metrics['f1'] * 100, 2),
            "f1_weighted": round(tm_metrics['f1_weighted'] * 100, 2),
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
            "f1_weighted": round(sm_metrics['f1_weighted'] * 100, 2),
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
    print(f"{'F1 Weighted (%)':<14} | {cr['TM']['f1_weighted']:>10.2f} | {cr['QM']['f1_weighted']:>10.2f} | {cr['SM']['f1_weighted']:>10.2f}")
    print(f"{'Sparsity (%)':<14} | {cr['TM']['sparsity']:>10.2f} | {cr['QM']['sparsity']:>10.2f} | {cr['SM']['sparsity']:>10.2f}")
    print(f"{'Size (MB)':<14} | {cr['TM']['size']:>10.2f} | {cr['QM']['size']:>10.2f} | {cr['SM']['size']:>10.2f}")


# In[6] DONE - VISUALIZATION FUNCTION WILL BE ADDED LATER." 

# To save the "results" dictionary, add this code at the end of the main.py (after all compression loops): 

# Save results with timestamp
now = datetime.now().strftime("%Y-%m-%d_%H-%M")
results_filename = os.path.join(experiment_folder, f"results_{MODEL_NAME}_{DATASET_NAME}_{now}.json")
with open(results_filename, 'w') as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {results_filename}")