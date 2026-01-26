"""
Flower client for federated learning with MedViTV2.

Each client:
1. Receives global model parameters from server
2. Trains on local BloodMNIST data
3. Returns updated parameters to server
"""

import warnings
warnings.filterwarnings("ignore")

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import NDArrays, Scalar

# Suppress Flower deprecation warnings
import logging
logging.getLogger("flwr").setLevel(logging.CRITICAL)

from .model import create_model, count_parameters
from .data import FederatedDataset


def compute_gradient_stats(initial_params: NDArrays, final_params: NDArrays) -> Dict:
    """
    Compute statistics about the weight delta (pseudo-gradient).

    In FL, clients send updated weights, not gradients directly.
    The "gradient" here is: delta = final_weights - initial_weights
    This represents the direction the client wants to move the model.
    """
    total_delta_norm = 0.0
    total_elements = 0
    all_deltas = []

    layer_stats = []

    for i, (init, final) in enumerate(zip(initial_params, final_params)):
        delta = final - init

        # Per-layer stats
        layer_norm = np.linalg.norm(delta)
        layer_mean = np.mean(delta)
        layer_std = np.std(delta)
        layer_max = np.max(np.abs(delta))

        layer_stats.append({
            "layer": i,
            "shape": delta.shape,
            "norm": layer_norm,
            "mean": layer_mean,
            "std": layer_std,
            "max_abs": layer_max,
        })

        total_delta_norm += layer_norm ** 2
        total_elements += delta.size
        all_deltas.append(delta.flatten())

    # Global stats
    all_deltas = np.concatenate(all_deltas)
    global_norm = np.sqrt(total_delta_norm)
    global_mean = np.mean(all_deltas)
    global_std = np.std(all_deltas)

    return {
        "global_norm": global_norm,
        "global_mean": global_mean,
        "global_std": global_std,
        "num_parameters": total_elements,
        "layer_stats": layer_stats,
    }


class MedViTClient(fl.client.NumPyClient):
    """
    Flower client for training MedViTV2 on BloodMNIST.
    """

    def __init__(
        self,
        client_id: int,
        fed_dataset: FederatedDataset,
        model_name: str = "medvit_small",
        num_classes: int = 8,
        img_size: int = 224,
        device: str = "auto",
        learning_rate: float = 1e-4,
        local_epochs: int = 1,
        batch_size: int = 32,
    ):
        """
        Args:
            client_id: Unique identifier for this client
            fed_dataset: Federated dataset manager
            model_name: Model architecture to use
            num_classes: Number of output classes
            img_size: Input image size
            device: Device to train on ("auto", "cpu", "cuda", "mps")
            learning_rate: Learning rate for local training
            local_epochs: Number of local epochs per round
            batch_size: Batch size for training
        """
        self.client_id = client_id
        self.fed_dataset = fed_dataset
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Create model (suppress verbose output)
        self.model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            img_size=img_size,
        ).to(self.device)

        # Get data loaders
        self.train_loader = fed_dataset.get_client_dataloader(
            client_id=client_id,
            batch_size=batch_size,
        )
        self.val_loader = fed_dataset.get_val_dataloader(batch_size=batch_size)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Training stats
        self.train_loss_history = []
        self.train_acc_history = []

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train model on local data.

        Args:
            parameters: Global model parameters from server
            config: Training configuration from server

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Store initial parameters (before training) to compute gradient later
        initial_parameters = [p.copy() for p in parameters]

        # Set model to received parameters
        self.set_parameters(parameters)

        # Get config values (with defaults)
        local_epochs = config.get("local_epochs", self.local_epochs)
        learning_rate = config.get("learning_rate", self.learning_rate)

        # Update learning rate if provided
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

        # Train
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(int(local_epochs)):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                epoch_correct += predicted.eq(labels).sum().item()
                epoch_total += labels.size(0)

            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total

        # Compute metrics
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        self.train_loss_history.append(avg_loss)
        self.train_acc_history.append(accuracy)

        # Get final parameters after training
        final_parameters = self.get_parameters(config={})

        # Compute gradient (weight delta) statistics
        grad_stats = compute_gradient_stats(initial_parameters, final_parameters)

        # Print training metrics and gradient info
        print(
            f"Client {self.client_id}: "
            f"loss={avg_loss:.4f}, acc={accuracy:.4f}, "
            f"samples={total}"
        )
        print(
            f"  └─ Gradient: norm={grad_stats['global_norm']:.6f}, "
            f"mean={grad_stats['global_mean']:.2e}, "
            f"std={grad_stats['global_std']:.2e}"
        )

        # Compute deltas (what we transmit for OTA)
        deltas = [final - init for final, init in zip(final_parameters, initial_parameters)]

        # Print first 3 delta values for first 3 layers
        print(f"  └─ Delta values (first 3 per layer):")
        for i in range(min(3, len(deltas))):
            vals = deltas[i].flatten()[:3]
            print(f"      Layer {i}: [{vals[0]:+.6f}, {vals[1]:+.6f}, {vals[2]:+.6f}, ...]")

        # Check if we should send deltas (for OTA) or full weights (standard FL)
        send_deltas = config.get("send_deltas", True)  # Default to deltas for OTA

        if send_deltas:
            # Return DELTAS - this is what gets transmitted over the air
            return (
                deltas,
                total,
                {
                    "loss": avg_loss,
                    "accuracy": accuracy,
                    "client_id": self.client_id,
                    "gradient_norm": float(grad_stats['global_norm']),
                    "gradient_mean": float(grad_stats['global_mean']),
                    "gradient_std": float(grad_stats['global_std']),
                    "is_delta": True,  # Flag so server knows this is a delta
                },
            )
        else:
            # Return full weights (standard FedAvg)
            return (
                final_parameters,
                total,
                {
                    "loss": avg_loss,
                    "accuracy": accuracy,
                    "client_id": self.client_id,
                    "gradient_norm": float(grad_stats['global_norm']),
                    "gradient_mean": float(grad_stats['global_mean']),
                    "gradient_std": float(grad_stats['global_std']),
                    "is_delta": False,
                },
            )

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate model on validation data.

        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, total, {
            "accuracy": accuracy,
            "client_id": self.client_id,
        }


def create_client_fn(
    fed_dataset: FederatedDataset,
    model_name: str = "medvit_small",
    num_classes: int = 8,
    img_size: int = 224,
    device: str = "auto",
    learning_rate: float = 1e-4,
    local_epochs: int = 1,
    batch_size: int = 32,
):
    """
    Create a client function for Flower simulation.

    Returns a function that creates a client given a client ID (cid).
    """

    def client_fn(cid: str) -> fl.client.NumPyClient:
        return MedViTClient(
            client_id=int(cid),
            fed_dataset=fed_dataset,
            model_name=model_name,
            num_classes=num_classes,
            img_size=img_size,
            device=device,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            batch_size=batch_size,
        )

    return client_fn


if __name__ == "__main__":
    # Test client creation
    print("Testing client creation...")

    # Create federated dataset
    fed_dataset = FederatedDataset(
        num_clients=3,
        partition="iid",
        img_size=224,
    )

    # Create a client
    client = MedViTClient(
        client_id=0,
        fed_dataset=fed_dataset,
        model_name="medvit_tiny",  # Use tiny for testing
        device="cpu",
    )

    # Test get_parameters
    params = client.get_parameters(config={})
    print(f"Number of parameter arrays: {len(params)}")

    # Test set_parameters
    client.set_parameters(params)
    print("Parameters set successfully")

    # Test a single fit round
    print("\nTesting fit...")
    updated_params, num_examples, metrics = client.fit(params, config={})
    print(f"Fit complete: {num_examples} examples, metrics={metrics}")

    # Test evaluate
    print("\nTesting evaluate...")
    loss, num_examples, metrics = client.evaluate(updated_params, config={})
    print(f"Evaluate complete: loss={loss:.4f}, metrics={metrics}")
