import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import OrderedDict
from federated_utils import get_model, prepare_datasets
from federated_utils import (
    MODEL_NAME, DATASET_NAME, LOCAL_EPOCH, BATCH_SIZE,
    OPTIMIZER, LEARNING_RATE, TOTAL_DEVICES_IN_THE_SYSTEM,  RESIZE_SIZE_FOR_VGG16, ALPHA)
import sys
class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        self.model_name = MODEL_NAME
        self.dataset_name = DATASET_NAME
        self.optimizer_name = OPTIMIZER
        self.learning_rate = LEARNING_RATE
        self.total_devices = TOTAL_DEVICES_IN_THE_SYSTEM
        self.batch_size = BATCH_SIZE
        self.alpha = ALPHA  # For non-IID data distribution

    def init_model_if_needed(self, config):
        if self.model is None:
            # model_name = config.get("model_name", "VGG16")
            # dataset_name = config.get("dataset_name", "CIFAR10")
            # model_name = MODEL_NAME
            # dataset_name = DATASET_NAME
            # optimizer_name = OPTIMIZER
            # learning_rate = LEARNING_RATE
            print(f"[CLIENT] Initializing model: {self.model_name} for dataset: {self.dataset_name}")
            device_loaders, _, input_channels, client_indices, train_dataset = prepare_datasets(self.dataset_name, self.total_devices, self.batch_size,  RESIZE_SIZE_FOR_VGG16, self.alpha)
            self.model = get_model(self.model_name, input_channels=input_channels).to(self.device)
            self.train_loader = device_loaders[self.ID]  # Each client uses a subset
            print(f"[CLIENT] Model {self.model_name} initialized for client {self.ID} on dataset {self.dataset_name}")
            # self.model_name = model_name
            # self.dataset_name = dataset_name

    def get_parameters(self, config):
        self.init_model_if_needed(config)
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.init_model_if_needed(config)
        self.set_parameters(parameters)
        self.model.train()
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # Use specified optimizer and LR
        if self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
                raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        

        dataset_size = len(self.train_loader.dataset)
        for epoch  in range(LOCAL_EPOCH):  # 1 local epoch
            print(f"[CLIENT] {self.model_name} on {self.dataset_name} - Starting epoch {epoch+1}")
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"[CLIENT] {self.model_name} - Epoch {epoch+1} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.4f} (dataset size: {dataset_size})")
        print(f"[CLIENT] {self.model_name} training completed. Returning parameters to server.")
        trained_params = self.get_parameters(config)
        checksum = sum([p.sum() for p in trained_params])
        print(f"[CLIENT][DEBUG] Sending weights checksum: {checksum:.6f}")
        return trained_params, len(self.train_loader.dataset), {}
        #return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        print(f"[CLIENT] Evaluating {self.model_name} on {self.dataset_name}")
        return 0.0, 0, {}


if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="SERVER_IP:8080", client=FLClient())

