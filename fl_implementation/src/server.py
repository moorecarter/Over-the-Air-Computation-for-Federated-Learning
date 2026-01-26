"""
Flower server for federated learning with OTA aggregation.

Coordinates the federated learning process:
1. Initialize global model
2. Distribute parameters to clients
3. Aggregate updates (with OTA channel simulation)
4. Evaluate global model
"""

from typing import Dict, List, Optional, Tuple, Callable
import torch
import numpy as np

import flwr as fl
from flwr.common import Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerConfig

from .model import create_model
from .ota_strategy import FedAvgOTA, create_ota_strategy


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average.

    Args:
        metrics: List of (num_examples, metrics_dict) tuples

    Returns:
        Aggregated metrics dictionary with per-client breakdown
    """
    # Collect accuracies and losses
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    total_examples = sum(examples)

    # Build per-client metrics
    per_client = {}
    for num_examples, m in metrics:
        client_id = m.get("client_id", "unknown")
        per_client[f"client_{client_id}_accuracy"] = m["accuracy"]
        per_client[f"client_{client_id}_loss"] = m.get("loss", 0)
        per_client[f"client_{client_id}_samples"] = num_examples
        # Gradient stats if available
        if "gradient_norm" in m:
            per_client[f"client_{client_id}_gradient_norm"] = m["gradient_norm"]
            per_client[f"client_{client_id}_gradient_mean"] = m["gradient_mean"]
            per_client[f"client_{client_id}_gradient_std"] = m["gradient_std"]

    return {
        "accuracy": sum(accuracies) / total_examples if total_examples > 0 else 0,
        "loss": sum(losses) / total_examples if total_examples > 0 else 0,
        **per_client,
    }


def create_evaluate_fn(
    model_name: str,
    num_classes: int,
    img_size: int,
    test_loader: torch.utils.data.DataLoader,
    device: str = "auto",
) -> Callable:
    """
    Create a server-side evaluation function.

    This evaluates the global model on a held-out test set.
    """
    # Set device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Create model for evaluation
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        img_size=img_size,
    ).to(device)

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on test set."""

        # Set parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        # Evaluate
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        print(f"\n[Round {server_round}] Global accuracy: {accuracy*100:.2f}%")

        return avg_loss, {"accuracy": accuracy}

    return evaluate


def create_fit_config_fn(
    local_epochs: int = 1,
    learning_rate: float = 1e-4,
    lr_decay: float = 0.99,
    send_deltas: bool = True,  # For OTA: send deltas instead of full weights
) -> Callable:
    """
    Create a function that returns training config for each round.

    Supports learning rate decay across rounds.

    Args:
        local_epochs: Number of local training epochs
        learning_rate: Initial learning rate
        lr_decay: Learning rate decay per round
        send_deltas: If True, clients send weight deltas (for OTA).
                     If False, clients send full weights (standard FL).
    """

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return training configuration for this round."""
        # Apply learning rate decay
        current_lr = learning_rate * (lr_decay ** (server_round - 1))

        return {
            "local_epochs": local_epochs,
            "learning_rate": current_lr,
            "server_round": server_round,
            "send_deltas": send_deltas,  # Tell clients to send deltas for OTA
        }

    return fit_config


def get_initial_parameters(
    model_name: str,
    num_classes: int,
    img_size: int,
) -> fl.common.Parameters:
    """
    Get initial model parameters for the server.

    Creates a fresh model and extracts its parameters.
    """
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        img_size=img_size,
    )

    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(ndarrays)


def create_server_strategy(
    model_name: str = "medvit_small",
    num_classes: int = 8,
    img_size: int = 224,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    device: str = "auto",
    # OTA parameters
    ota_channel: str = "ideal",
    snr_db: Optional[float] = None,
    fading_type: Optional[str] = None,
    # FL parameters
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    local_epochs: int = 1,
    learning_rate: float = 1e-4,
    lr_decay: float = 0.99,
) -> FedAvgOTA:
    """
    Create a server strategy with OTA aggregation.

    Args:
        model_name: Model architecture name
        num_classes: Number of output classes
        img_size: Input image size
        test_loader: DataLoader for server-side evaluation
        device: Device for evaluation
        ota_channel: OTA channel preset ("ideal", "good", "moderate", etc.)
        snr_db: Override SNR (dB)
        fading_type: Override fading type
        fraction_fit: Fraction of clients to use for training
        fraction_evaluate: Fraction of clients to use for evaluation
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum available clients to start round
        local_epochs: Number of local epochs per round
        learning_rate: Initial learning rate
        lr_decay: Learning rate decay per round

    Returns:
        Configured FedAvgOTA strategy
    """
    # Get initial parameters
    initial_parameters = get_initial_parameters(model_name, num_classes, img_size)

    # Create evaluation function if test loader provided
    evaluate_fn = None
    if test_loader is not None:
        evaluate_fn = create_evaluate_fn(
            model_name=model_name,
            num_classes=num_classes,
            img_size=img_size,
            test_loader=test_loader,
            device=device,
        )

    # Create fit config function
    fit_config_fn = create_fit_config_fn(
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
    )

    # Build OTA channel kwargs
    ota_kwargs = {}
    if snr_db is not None:
        ota_kwargs["snr_db"] = snr_db
    if fading_type is not None:
        ota_kwargs["fading_type"] = fading_type

    # Create strategy with preset, then override with any explicit params
    strategy = create_ota_strategy(
        channel_config=ota_channel,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config_fn,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        **ota_kwargs,
    )

    return strategy
