#!/usr/bin/env python3
"""
Federated Learning Demo with MedViTV2, BloodMNIST, and OTA Aggregation.
"""

import argparse
import json
import logging
import os
import sys
import warnings
import threading
import time



# ============== AGGRESSIVE LOGGING SUPPRESSION ==============
# Must happen BEFORE any other imports

# Suppress all Python warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Environment variables to suppress various loggers
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["ABSL_MIN_LOG_LEVEL"] = "4"  # Suppress abseil C++ logs (0=INFO,1=WARNING,2=ERROR,3=FATAL,4=OFF)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Completely disable Ray's logging
os.environ["RAY_LOG_TO_STDERR"] = "0"

# Set root logger to only show CRITICAL
logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
logging.getLogger().setLevel(logging.CRITICAL)

# Suppress specific loggers
_loggers_to_suppress = [
    "flwr", "flower", "grpc", "urllib3", "asyncio", "absl", "werkzeug",
    "numba", "PIL", "matplotlib", "tensorflow", "torch", "torchvision"
]
for logger_name in _loggers_to_suppress:
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(logging.CRITICAL)
    _logger.disabled = True
    _logger.handlers = []
    _logger.propagate = False

from datetime import datetime
from pathlib import Path

# Suppress flwr import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import flwr as fl
    from flwr.server import ServerConfig

# Re-suppress after imports
for logger_name in _loggers_to_suppress:
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(logging.CRITICAL)
    _logger.disabled = True
    _logger.handlers = []

from src.data import FederatedDataset
from src.client import create_client_fn
from src.server import create_server_strategy
from src.model import create_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Federated Learning with MedViTV2 and OTA Aggregation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="medvit_tiny",
        choices=["medvit_tiny", "medvit_small", "medvit_base", "medvit_small_no_kan"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=28,
        choices=[28, 64, 128, 224],
        help="Input image size (28 is fastest for demo)",
    )

    # Federated learning configuration
    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
        help="Number of federated clients",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Number of federated learning rounds",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Number of local training epochs per round",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    # Data partitioning
    parser.add_argument(
        "--partition",
        type=str,
        default="iid",
        choices=["iid", "non_iid", "pathological"],
        help="Data partitioning strategy",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha for non_iid partition (lower = more heterogeneous)",
    )

    # OTA channel configuration
    parser.add_argument(
        "--ota-channel",
        type=str,
        default="ideal",
        choices=["ideal", "good", "moderate", "challenging", "harsh"],
        help="OTA channel condition preset",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=None,
        help="Override SNR in dB (optional)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training",
    )

    # Experiment tracking
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (for logging)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save results",
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # USRP configuration
    parser.add_argument(
        "--use-usrp",
        action="store_true",
        help="Use real USRP hardware instead of simulated OTA channel",
    )
    parser.add_argument(
        "--server-usrp-addr",
        type=str,
        default=None,
        help="IP address of the server USRP (RX)",
    )
    parser.add_argument(
        "--client-usrp-addrs",
        nargs="+",
        type=str,
        default=None,
        help="IP addresses of client USRPs (TX), e.g. 192.168.40.3 192.168.40.4",
    )

    return parser.parse_args()


def main():
    """Run the federated learning experiment."""
    args = parse_args()

    # Set experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"fl_{args.model}_{args.partition}_{args.ota_channel}_{timestamp}"

    print(f"FL: {args.num_clients} clients, {args.num_rounds} rounds, {args.model} ({args.img_size}px), USRP={'ON' if args.use_usrp else 'OFF'}")

    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    model = create_model(
        model_name=args.model,
        num_classes=8,  # BloodMNIST has 8 classes
        img_size=args.img_size,
    )
    # print(f"Model: {args.model} ({count_parameters(model):,} parameters)")
    del model  # Free memory

    # Create federated dataset
    fed_dataset = FederatedDataset(
        num_clients=args.num_clients,
        partition=args.partition,
        alpha=args.alpha,
        img_size=args.img_size,
        seed=args.seed,
    )

    # Create client function
    client_fn = create_client_fn(
        fed_dataset=fed_dataset,
        model_name=args.model,
        num_classes=8,
        img_size=args.img_size,
        device=args.device,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
    )

    # Create server strategy
    strategy = create_server_strategy(
        server_usrp_addr=args.server_usrp_addr if args.use_usrp else None,
        client_usrp_addrs=args.client_usrp_addrs if args.use_usrp else None,
        model_name=args.model,
        num_classes=8,
        img_size=args.img_size,
        test_loader=fed_dataset.get_test_dataloader(batch_size=args.batch_size),
        device=args.device,
        ota_channel=args.ota_channel,
        snr_db=args.snr_db,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
    )

    # Run Training

    # Launch clients in a background thread, then run server on main thread
    # (Flower's server needs main thread for signal.signal() handlers)
    GRPC_MAX_MESSAGE_LENGTH = 512 * 1024 * 1024  # 512 MB
    client_threads = []

    def start_clients():
        time.sleep(2)  # let server bind first
        for i in range(args.num_clients):
            client = client_fn(str(i))
            t = threading.Thread(
                target=fl.client.start_client,
                kwargs={
                    "server_address": "localhost:8080",
                    "client": client.to_client(),
                    "grpc_max_message_length": GRPC_MAX_MESSAGE_LENGTH,
                }
            )
            t.start()
            client_threads.append(t)
        for t in client_threads:
            t.join()

    clients_thread = threading.Thread(target=start_clients)
    clients_thread.start()

    # Run server on main thread
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        grpc_max_message_length=GRPC_MAX_MESSAGE_LENGTH,
    )

    clients_thread.join()



    # Save results

    # Get training metrics (fit) if available
    metrics_fit = {}
    if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
        metrics_fit = dict(history.metrics_distributed_fit)

    results = {
        "losses_distributed": history.losses_distributed,
        "losses_centralized": history.losses_centralized,
        "metrics_distributed": dict(history.metrics_distributed),
        "metrics_distributed_fit": metrics_fit,  # Training metrics per client
        "metrics_centralized": dict(history.metrics_centralized),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Final summary
    if history.metrics_centralized.get("accuracy"):
        final_acc = history.metrics_centralized["accuracy"][-1][1]
        init_acc = history.metrics_centralized["accuracy"][0][1]
        improvement = final_acc - init_acc
        print(f"\nFinal: {init_acc*100:.1f}% -> {final_acc*100:.1f}% (+{improvement*100:.1f}%)")

    print(f"Saved to: {results_path}")


if __name__ == "__main__":
    main()
