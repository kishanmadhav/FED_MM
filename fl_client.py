import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset
from model_and_data import PneumoniaCNN, RSNAPneumoniaDataset
import sys
import random
import copy
import socket
import pickle
import torch.nn.functional as F
import torch.nn as nn

CSV_PATH = r"C:\CDAC\stage_2_train_labels.csv"
IMG_DIR = r"C:\CDAC\train_preprocess"

# Each client uses a different data split

def get_client_dataset(client_id):
    dataset = RSNAPneumoniaDataset(CSV_PATH, IMG_DIR)
    total = len(dataset)
    
    # Unequal split ratios (40%, 35%, 25%)
    if client_id == "1":
        start_idx = 0
        end_idx = int(0.4 * total)  # 40% of data
    elif client_id == "2":
        start_idx = int(0.4 * total)
        end_idx = int(0.75 * total)  # Next 35% of data
    else:
        start_idx = int(0.75 * total)
        end_idx = total  # Remaining 25% of data
    
    indices = range(start_idx, end_idx)
    subset = Subset(dataset, indices)
    print("Client {} initialized with {:.1f}% of total data ({} samples)".format(
        client_id, 100 * len(subset) / total, len(subset)))
    
    # Display architecture ID based on data characteristics
    data_size = len(subset)
    positive_samples = sum(dataset.targets[i] == 1 for i in indices)
    arch_id = "C{}_S{}_P{:.2f}".format(
        client_id, 
        data_size,
        positive_samples / data_size
    )
    print("Client {} Architecture ID: {}".format(client_id, arch_id))
    
    return subset

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.dataset = get_client_dataset(client_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate data characteristics for adaptive architecture
        data_size = len(self.dataset)
        positive_samples = sum(1 for i in range(len(self.dataset)) 
                             if self.dataset.dataset.targets[self.dataset.indices[i]] == 1)
        positive_ratio = positive_samples / data_size
        
        # Configure adaptive architecture
        adaptive_config = {
            "data_size": data_size,
            "positive_ratio": positive_ratio,
            "client_id": client_id,
            "device_type": str(self.device)
        }
        
        # Initialize model with adaptive configuration
        self.model = PneumoniaCNN(adaptive_config=adaptive_config)
        self.model.to(self.device)
        
        # Create architecture ID based on model and data characteristics
        model_params = sum(p.numel() for p in self.model.parameters())
        arch_id = {
            "client_id": client_id,
            "data_size": data_size,
            "positive_ratio": "{:.3f}".format(positive_ratio),
            "model_params": "{:.2f}M".format(model_params/1e6),
            "backbone": self.model.model_name,
            "device": str(self.device),
            "classifier_depth": len([m for m in self.model.classifier if isinstance(m, nn.Linear)])
        }
        
        print("\nArchitecture Details for Client {}:".format(client_id))
        print("----------------------------------------")
        print("Data Size: {} samples".format(arch_id['data_size']))
        print("Positive Class Ratio: {}".format(arch_id['positive_ratio']))
        print("Model Parameters: {}".format(arch_id['model_params']))
        print("Backbone: {}".format(arch_id['backbone']))
        print("Device: {}".format(arch_id['device']))
        print("Classifier Depth: {} layers".format(arch_id['classifier_depth']))
        print("----------------------------------------\n")
        
        self.arch_id = arch_id

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        round_number = config.get("round_number", 0)
        print("\nStarting training round {} for Client {}".format(round_number, self.client_id))
        print("Using adaptive architecture:", self.arch_id)
        
        self.set_parameters(parameters)
        self.model.train()
        
        # Adjust training parameters based on data size
        if self.arch_id['data_size'] < 500:
            batch_size = 8
            epochs = 4  # More epochs for smaller datasets
        elif self.arch_id['data_size'] < 1000:
            batch_size = 16
            epochs = 3
        else:
            batch_size = 32
            epochs = 2  # Fewer epochs for larger datasets
        
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Adaptive learning rate based on model size
        base_lr = 0.001
        if self.model.model_name == 'efficientnet_b0':
            max_lr = base_lr * 1.5
        elif self.model.model_name == 'efficientnet_b1':
            max_lr = base_lr
        else:
            max_lr = base_lr * 0.75
        
        # Use AdamW with weight decay and learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=0.01)
        scheduler = torch.optim.OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(loader),
            pct_start=0.3
        )
        
        # Get class weights from dataset
        class_weights = self.dataset.dataset.get_class_weights()
        class_weights = class_weights.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        print("Using class weights:", class_weights.cpu().numpy())
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Use mixed precision for better performance
                with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / len(loader)
            print("Epoch {}/{}, Loss: {:.4f}".format(epoch+1, epochs, avg_epoch_loss))
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print("\nClient {} completed round {} with average loss: {:.4f}".format(
            self.client_id, round_number, avg_loss))
        
        return self.get_parameters(config={}), len(self.dataset), {
            "loss": avg_loss,
            "client_id": self.client_id,
            "round_number": round_number,
            "arch_id": self.arch_id
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        loader = DataLoader(self.dataset, batch_size=32, shuffle=False)
        
        # Get class weights from dataset
        class_weights = loader.dataset.get_class_weights()
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item() * images.size(0)
                
                # Use adjusted threshold for predictions
                probabilities = F.softmax(outputs, dim=1)
                predicted = (probabilities[:, 1] > PREDICTION_THRESHOLD).long()
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        
        print("Client {} evaluation - Accuracy: {:.2f}%, Loss: {:.4f}".format(
            self.client_id, accuracy, avg_loss))
        
        return float(avg_loss), len(self.dataset), {
            "accuracy": accuracy,
            "client_id": self.client_id
        }

def local_train_scaffold(model, dataloader, optimizer, loss_fn, global_c, ci, device, local_epochs=1, lr=0.001):
    model.train()
    for epoch in range(local_epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            # SCAFFOLD correction: add control variate difference
            correction = 0.0
            for name, param in model.named_parameters():
                correction += torch.sum((global_c[name] - ci[name]) * param)
            loss += correction
            loss.backward()
            optimizer.step()
    # After training, update ci
    new_ci = copy.deepcopy(ci)
    for name, param in model.named_parameters():
        new_ci[name] = ci[name] - global_c[name] + (global_c[name] - ci[name])
    return model.state_dict(), new_ci

def init_control_variate(model):
    return {k: torch.zeros_like(v) for k, v in model.state_dict().items()}

def add_state_dicts(a, b):
    return {k: a[k] + b[k] for k in a}

def sub_state_dicts(a, b):
    return {k: a[k] - b[k] for k in a}

def mul_state_dict(s, scalar):
    return {k: v * scalar for k, v in s.items()}

def average_state_dicts(state_dicts, weights):
    total_weight = sum(weights)
    avg = {}
    for k in state_dicts[0]:
        avg[k] = sum(w * sd[k] for sd, w in zip(state_dicts, weights)) / total_weight
    return avg

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    # Get class weights from dataset
    class_weights = dataloader.dataset.get_class_weights()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    with torch.no_grad():
        for images, labels in dataloader:
            # Ensure inputs are float32
            images = images.float().to(device)
            labels = labels.to(device)
            
            # Use mixed precision for forward pass
            with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            
            # Use adjusted threshold for predictions
            probabilities = F.softmax(outputs.float(), dim=1)  # Convert to float32 for softmax
            predicted = (probabilities[:, 1] > PREDICTION_THRESHOLD).long()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy, avg_loss

if __name__ == "__main__":
    client_id = sys.argv[1]  # Pass 1, 2, or 3 as argument
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(client_id)) 