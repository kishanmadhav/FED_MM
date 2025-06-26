import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import pydicom
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, random_split, Subset
import torchvision.models as models
from torchvision import transforms
from torch.cuda.amp import autocast
import time
import torch.optim as optim
from collections import OrderedDict
from typing import Dict, List, Tuple
import json
from torch.utils.data import DataLoader
import logging
import hashlib
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PneumoniaCNN(nn.Module):
    """Enhanced CNN model for pneumonia detection with adaptive architecture"""
    def __init__(self, model_name=None, num_classes=2, pretrained=True, adaptive_config=None):
        super(PneumoniaCNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("✓ Using device: {}".format(self.device))
        
        self.model_name = model_name
        self.adaptive_config = adaptive_config or {}
        
        # Get model type from config
        model_type = self.adaptive_config.get('model_type', 'densenet121')
        self.model_name = model_type
        
        # Set architecture parameters based on model type
        if model_type == 'mobilenet_v2':
            self.dropout_rate = 0.3
            self.feature_dim = 1280
        elif model_type == 'resnet18':
            self.dropout_rate = 0.4
            self.feature_dim = 512
        else:  # densenet121
            self.dropout_rate = 0.5
            self.feature_dim = 1024
        
        # Initialize backbone based on model type
        if model_type == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            # Modify first layer for single channel
            self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            
        elif model_type == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            # Modify first layer for single channel
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        else:  # densenet121
            self.backbone = models.densenet121(pretrained=pretrained)
            # Modify first layer for single channel
            self.backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the original classifier
        if model_type == 'mobilenet_v2':
            self.backbone.classifier = nn.Identity()
        elif model_type == 'resnet18':
            self.backbone.fc = nn.Identity()
        else:  # densenet121
            self.backbone.classifier = nn.Identity()
        
        # Build consistent classifier architecture for all models
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Print architecture summary
        print("\nModel Architecture Configuration:")
        print("- Model Type: {}".format(model_type))
        print("- Dropout Rate: {:.1f}".format(self.dropout_rate))
        print("- Feature Dimension: {}".format(self.feature_dim))
    
    def forward(self, x):
        if self.model_name == 'mobilenet_v2':
            features = self.backbone.features(x)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
        elif self.model_name == 'resnet18':
            features = self.backbone.conv1(x)
            features = self.backbone.bn1(features)
            features = self.backbone.relu(features)
            features = self.backbone.maxpool(features)
            features = self.backbone.layer1(features)
            features = self.backbone.layer2(features)
            features = self.backbone.layer3(features)
            features = self.backbone.layer4(features)
            features = self.backbone.avgpool(features)
            features = torch.flatten(features, 1)
        else:  # densenet121
            features = self.backbone.features(x)
            features = F.relu(features, inplace=True)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
        
        return self.classifier(features)

class ResidualBlock(nn.Module):
    """Enhanced Residual block with bottleneck architecture"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.bottleneck = channels // 4
        
        self.block = nn.Sequential(
            # Bottleneck down
            nn.Linear(channels, self.bottleneck),
            nn.BatchNorm1d(self.bottleneck),
            nn.ReLU(),
            
            # Processing at reduced dimensionality
            nn.Linear(self.bottleneck, self.bottleneck),
            nn.BatchNorm1d(self.bottleneck),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Bottleneck up
            nn.Linear(self.bottleneck, channels),
            nn.BatchNorm1d(channels)
        )
        
        # Skip connection with gating
        self.gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        # Gated skip connection
        gate_value = self.gate(identity)
        out = gate_value * out + (1 - gate_value) * identity
        return self.relu(out)

class ParallelBlock(nn.Module):
    """Parallel processing block with multiple paths"""
    def __init__(self, in_channels, out_channels):
        super(ParallelBlock, self).__init__()
        
        self.path1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        self.path2 = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        self.path3 = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        self.combine = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        # Concatenate the parallel paths
        combined = torch.cat([p1, p2, p3], dim=1)
        return self.combine(combined)

class PositiveBiasLinear(nn.Linear):
    """Linear layer with built-in positive bias for pneumonia detection"""
    def __init__(self, in_features, out_features, bias=True, bias_factor=0.1):
        super(PositiveBiasLinear, self).__init__(in_features, out_features, bias)
        self.bias_factor = bias_factor
    
    def forward(self, x):
        output = super().forward(x)
        # Add a small positive bias to the pneumonia class logit
        if output.size(1) == 2:  # Binary classification
            output[:, 1] = output[:, 1] + self.bias_factor
        return output

class RSNAPneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, augment=False, max_retries=5, retry_delay=2):
        self.pneumonia_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.augment = augment
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize device first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Dataset initialized on device: {}".format(self.device))
        
        # Enhanced transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380, 380)),  # Larger input size for better feature extraction
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ]) if transform is None else transform
        
        # Enhanced augmentation pipeline
        self.augment_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.12, 0.12),
                scale=(0.85, 1.15),
                shear=10
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.3),
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        self._create_labels()
        self._verify_image_files()
        self._calculate_class_weights()
    
    def _calculate_class_weights(self):
        labels = self.pneumonia_frame['Target'].values
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        # Adjusted weight calculation with focal loss consideration
        self.class_weights = torch.FloatTensor(np.sqrt(total_samples / (class_counts * class_counts)))
        self.class_weights = self.class_weights.to(self.device)
    
    def get_class_weights(self):
        return self.class_weights
    
    def _create_labels(self):
        if 'Target' in self.pneumonia_frame.columns:
            print("✓ Using 'Target' column for labels")
        else:
            bbox_cols = ['x', 'y', 'width', 'height']
            if all(col in self.pneumonia_frame.columns for col in bbox_cols):
                self.pneumonia_frame['Target'] = (
                    (self.pneumonia_frame[bbox_cols].fillna(0) != 0).any(axis=1)
                ).astype(int)
                print("✓ Created labels from bounding box data")
            else:
                print("⚠ Warning: Creating default labels (review your CSV structure)")
                self.pneumonia_frame['Target'] = 0

        if 'Target' in self.pneumonia_frame.columns:
            class_counts = self.pneumonia_frame['Target'].value_counts().sort_index()
            print("Class distribution:")
            print("  Normal (0): {} samples".format(class_counts.get(0, 0)))
            print("  Pneumonia (1): {} samples".format(class_counts.get(1, 0)))
    
    def _verify_image_files(self):
        print("Verifying image files...")
        sample_size = min(10, len(self.pneumonia_frame))
        found_extensions = set()

        for i in range(sample_size):
            patient_id = self.pneumonia_frame.iloc[i]['patientId']
            extensions = ['.dcm', '.png', '.jpg', '.jpeg']
            for ext in extensions:
                img_path = os.path.join(self.root_dir, patient_id + ext)
                if os.path.exists(img_path):
                    found_extensions.add(ext)
                    break

        if found_extensions:
            self.common_extension = list(found_extensions)[0]
            print("✓ Found images with extension: {}".format(self.common_extension))
        else:
            print("⚠ Warning: No sample images found, defaulting to .dcm")
            self.common_extension = '.dcm'
    
    def __len__(self):
        return min(150, len(self.pneumonia_frame))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        patient_id = self.pneumonia_frame.iloc[idx]['patientId']
        img_path = os.path.join(self.root_dir, patient_id + self.common_extension)
        
        if not os.path.exists(img_path):
            extensions = ['.dcm', '.png', '.jpg', '.jpeg']
            for ext in extensions:
                test_path = os.path.join(self.root_dir, patient_id + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break
        
        image = torch.zeros(1, 380, 380)  # Default empty image
        
        for attempt in range(self.max_retries):
            try:
                if img_path.endswith('.dcm'):
                    dicom = pydicom.dcmread(img_path)
                    image = dicom.pixel_array.astype(np.float32)
                else:
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        image = np.zeros((380, 380), dtype=np.uint8)
                    image = image.astype(np.float32)
                
                if image.max() > 10:
                    image = image / 255.0
                
                # Apply augmentation if enabled
                if self.augment and np.random.rand() > 0.5:
                    image = self.augment_transforms(image)
                else:
                    image = self.transform(image)
                
                break  # Successfully loaded image, break the retry loop

            except Exception as e:
                if attempt == self.max_retries - 1:  # Last attempt
                    print("Error loading image {} after {} attempts: {}".format(
                        img_path, self.max_retries, str(e)))
                else:
                    time.sleep(self.retry_delay)  # Wait before retrying
        
        label = int(self.pneumonia_frame.iloc[idx]['Target'])
        return image.to(self.device), torch.tensor(label, device=self.device)

def get_train_test_datasets(client_data, num_samples=150):
    # Assuming client_data is a list of datasets for each client
    train_data = []
    test_data = []
    
    for data in client_data:
        # Randomly sample 150 samples from the client's data
        sampled_data = random.sample(data, min(num_samples, len(data)))
        train_data.extend(sampled_data)  # Add to training data
        # You can also define how to split test_data if needed

    return train_data, test_data

def get_client_dataset(dataset, client_id, num_clients):
    """Split dataset into equal parts for each client."""
    total_size = len(dataset)
    per_client_size = total_size // num_clients
    start_idx = (client_id - 1) * per_client_size
    end_idx = start_idx + per_client_size if client_id < num_clients else total_size
    
    indices = list(range(start_idx, end_idx))
    return Subset(dataset, indices)

def get_test_dataset(csv_file, root_dir, test_ratio=0.2, seed=42):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    
    full_dataset = RSNAPneumoniaDataset(csv_file, root_dir, augment=False)
    total = len(full_dataset)
    test_size = int(test_ratio * total)
    train_size = total - test_size
    
    generator = torch.Generator().manual_seed(seed)
    _, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=generator
    )
    return test_dataset 

class AdaptiveBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class AdaptivePneumoniaCNN(nn.Module):
    def __init__(self, architecture_config: Dict):
        super().__init__()
        self.config = architecture_config
        self.features = self._make_layers()
        self.classifier = self._make_classifier()
        self.architecture_id = self._generate_architecture_id()

    def _make_layers(self):
        layers = []
        in_channels = 1  # Grayscale X-ray images
        
        for i, filters in enumerate(self.config['filters']):
            layers.append(
                AdaptiveBlock(
                    in_channels, 
                    filters, 
                    self.config['kernel_sizes'][i]
                )
            )
            in_channels = filters
            
        return nn.Sequential(*layers)

    def _make_classifier(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.config['filters'][-1], 2)  # Binary classification
        )

    def _generate_architecture_id(self):
        """Generate unique identifier for this architecture"""
        return hash(json.dumps(self.config, sort_keys=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ArchitectureManager:
    def __init__(self, client_id: str, device: torch.device):
        self.client_id = client_id
        self.device = device
        self.current_architecture = None
        self.performance_history = []
        self.resource_monitor = ResourceMonitor()
        
    def analyze_data_complexity(self, dataset: DataLoader) -> float:
        """Analyze the complexity of client's dataset"""
        complexity_score = 0.0
        feature_stats = []
        
        # Collect statistics from a batch of data
        for batch, _ in dataset:
            # Calculate feature statistics
            feature_stats.append({
                'mean': torch.mean(batch).item(),
                'std': torch.std(batch).item(),
                'entropy': self._calculate_entropy(batch)
            })
            
            if len(feature_stats) >= 5:  # Limit analysis to 5 batches
                break
                
        # Aggregate statistics
        complexity_score = np.mean([
            stats['entropy'] * stats['std'] 
            for stats in feature_stats
        ])
        
        return complexity_score

    def _calculate_entropy(self, batch: torch.Tensor) -> float:
        """Calculate entropy of the input batch"""
        histogram = torch.histc(batch, bins=20)
        histogram = histogram / histogram.sum()
        entropy = -torch.sum(histogram * torch.log2(histogram + 1e-7))
        return entropy.item()

    def generate_architecture(self, complexity_score: float, 
                            available_memory: float) -> Dict:
        """Generate architecture based on data complexity and resources"""
        base_filters = int(32 * (1 + complexity_score))
        num_layers = min(4, int(2 + complexity_score))
        
        architecture = {
            'filters': [
                min(base_filters * (2**i), 512) 
                for i in range(num_layers)
            ],
            'kernel_sizes': [
                3 if i < 2 else 5 
                for i in range(num_layers)
            ]
        }
        
        # Adjust based on available memory
        while self._estimate_model_size(architecture) > available_memory:
            architecture = self._reduce_architecture(architecture)
            
        return architecture

    def _estimate_model_size(self, architecture: Dict) -> float:
        """Estimate model size in MB"""
        total_params = 0
        in_channels = 1
        
        for i, filters in enumerate(architecture['filters']):
            kernel_size = architecture['kernel_sizes'][i]
            params = (in_channels * filters * kernel_size * kernel_size) + filters
            total_params += params
            in_channels = filters
            
        # Convert to MB (assuming 32-bit floats)
        return (total_params * 4) / (1024 * 1024)

    def _reduce_architecture(self, architecture: Dict) -> Dict:
        """Reduce architecture size while maintaining proportions"""
        new_architecture = {
            'filters': [f // 2 for f in architecture['filters']],
            'kernel_sizes': architecture['kernel_sizes']
        }
        return new_architecture

class ResourceMonitor:
    def __init__(self):
        self.memory_threshold = 0.9  # 90% of available memory
        
    def get_available_memory(self) -> float:
        """Get available GPU/CPU memory in MB"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        else:
            import psutil
            return psutil.virtual_memory().available / (1024 * 1024)

    def check_resources(self) -> Dict:
        return {
            'available_memory': self.get_available_memory(),
            'memory_threshold': self.memory_threshold
        }

class AdaptiveFederatedClient:
    def __init__(self, client_id: str, dataset: torch.utils.data.Dataset):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.arch_manager = ArchitectureManager(client_id, self.device)
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
    def initialize_model(self):
        """Initialize or update model architecture"""
        # Analyze data and resources
        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        complexity_score = self.arch_manager.analyze_data_complexity(dataloader)
        resources = self.arch_manager.resource_monitor.check_resources()
        
        # Generate appropriate architecture
        architecture = self.arch_manager.generate_architecture(
            complexity_score,
            resources['available_memory']
        )
        
        # Initialize model
        self.model = AdaptivePneumoniaCNN(architecture).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        logger.info("Client {} initialized with architecture: {}".format(self.client_id, architecture))
        
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }

class AdaptiveFederatedServer:
    def __init__(self):
        self.global_model = None
        self.client_architectures = {}
        self.round = 0
        
    def aggregate_models(self, client_updates: List[Tuple[AdaptivePneumoniaCNN, float]]) -> Dict:
        """Aggregate updates from heterogeneous client models"""
        if not client_updates:
            return None
            
        # Group models by architecture
        architecture_groups = self._group_by_architecture(client_updates)
        
        # Aggregate within each architecture group
        aggregated_groups = {}
        for arch_id, group in architecture_groups.items():
            if group:
                aggregated_groups[arch_id] = self._aggregate_same_architecture(group)
                
        # Store aggregated models
        self.client_architectures = aggregated_groups
        
        with round_condition:
            # Wait until the round is ready for this client
            while True:
                # Only assign if this client hasn't submitted for this round
                if self.client_id not in client_updates[self.round]:
                    break
                # Otherwise, wait for the next round
                round_condition.wait()
            # Now send assignment
        
        return aggregated_groups
    
    def _group_by_architecture(self, 
                             client_updates: List[Tuple[AdaptivePneumoniaCNN, float]]) -> Dict:
        """Group client updates by architecture"""
        groups = {}
        for model, weight in client_updates:
            arch_id = model.architecture_id
            if arch_id not in groups:
                groups[arch_id] = []
            groups[arch_id].append((model, weight))
        return groups
    
    def _aggregate_same_architecture(self, 
                                   group: List[Tuple[AdaptivePneumoniaCNN, float]]) -> OrderedDict:
        """Aggregate models with the same architecture"""
        total_weight = sum(weight for _, weight in group)
        averaged_state = OrderedDict()
        
        # Get the state dict of the first model
        first_state = group[0][0].state_dict()
        
        for key in first_state.keys():
            averaged_state[key] = sum(
                model.state_dict()[key] * (weight / total_weight)
                for model, weight in group
            )
            
        return averaged_state

def get_model_hash(model):
    # Simple hash of all parameters for quick comparison
    params = b''.join([p.detach().cpu().numpy().tobytes() for p in model.parameters()])
    return hashlib.md5(params).hexdigest()

def main():
    # Initialize server
    server = AdaptiveFederatedServer()
    
    # Initialize clients
    clients = []
    for i in range(3):  # 3 clients
        client_dataset = get_client_dataset(str(i+1))  # Your existing dataset split function
        client = AdaptiveFederatedClient(str(i+1), client_dataset)
        client.initialize_model()
        clients.append(client)
    
    # Training loop
    num_rounds = 20
    for round in range(num_rounds):
        logger.info("Starting round {}".format(round+1))
        
        # Client updates
        client_updates = []
        for client in clients:
            # Train client model
            dataloader = DataLoader(client.dataset, batch_size=32, shuffle=True)
            metrics = client.train_epoch(dataloader)
            
            # Collect update
            client_updates.append((client.model, len(client.dataset)))
            
            logger.info("Client {} - Loss: {:.4f}, Accuracy: {:.2f}%".format(
                client.client_id, metrics['loss'], metrics['accuracy']))
        
        # Server aggregation
        aggregated_models = server.aggregate_models(client_updates)
        
        # Update clients with appropriate aggregated model
        for client in clients:
            arch_id = client.model.architecture_id
            if arch_id in aggregated_models:
                client.model.load_state_dict(aggregated_models[arch_id])
                
        logger.info("Completed round {}".format(round+1))

def train_model(model, dataset, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    batch_num = 0
    for images, labels in loader:
        batch_num += 1
        logger.info("[Client] Training batch {}".format(batch_num))
        try:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        except Exception as e:
            logger.error("[Client] Error in training batch: {}".format(str(e)))
            continue

    if batch_num == 0:
        logger.error("[Client] No batches processed! Check your dataset and DataLoader.")
    # Calculate metrics
    avg_loss = total_loss / batch_num if batch_num > 0 else float('nan')
    accuracy = 100. * correct / total if total > 0 else 0.0
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }

if __name__ == "__main__":
    main() 