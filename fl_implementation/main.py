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
import contextlib
import io

# ============== AGGRESSIVE LOGGING SUPPRESSION ==============
# Must happen BEFORE any other imports

# Suppress all Python warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Environment variables to suppress various loggers
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEDUP_LOGS_AGG_WINDOW_S"] = "9999"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

# Completely disable Ray's logging
os.environ["RAY_LOG_TO_STDERR"] = "0"

# Set root logger to only show CRITICAL
logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
logging.getLogger().setLevel(logging.CRITICAL)

# Suppress specific loggers
_loggers_to_suppress = [
    "ray", "ray.tune", "ray.rllib", "ray.train", "ray.data", "ray.serve",
    "ray.air", "ray._private", "ray.worker", "ray.util",
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
from src.model import create_model, count_parameters


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

    return parser.parse_args()


def main():
    """Run the federated learning experiment."""
    args = parse_args()

    # Set experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"fl_{args.model}_{args.partition}_{args.ota_channel}_{timestamp}"

    print("=" * 60)
    print("Federated Learning Demo")
    print("=" * 60)
    print(f"Config: {args.num_clients} clients, {args.num_rounds} rounds, {args.partition} partition, {args.ota_channel} channel")

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
    print(f"Model: {args.model} ({count_parameters(model):,} parameters)")
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

    # Run simulation
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    # Create a filter to capture only client output and clean Ray prefixes
    class OutputFilter:
        def __init__(self, stream):
            self.stream = stream
            self.keep_patterns = ["Client ", "└─", "Layer ", "[Round", "Global accuracy"]
            self.skip_patterns = ["raylet", "gcs_server", "MallocStack", "metrics", "core_worker", "file_system_monitor"]

        def write(self, text):
            for line in text.splitlines(keepends=True):
                # Skip lines with unwanted patterns
                if any(p in line for p in self.skip_patterns):
                    continue
                # Keep lines with wanted patterns
                if any(p in line for p in self.keep_patterns):
                    # Remove Ray prefix: [36m(ClientAppActor pid=...)[0m
                    import re
                    clean = re.sub(r'\x1b\[\d+m\([^)]+\)\x1b\[0m\s*', '', line)
                    if not clean.endswith('\n'):
                        clean += '\n'
                    self.stream.write(clean)

        def flush(self):
            self.stream.flush()

    # Suppress Ray stderr and filter stdout
    import ray
    if ray.is_initialized():
        ray.shutdown()

    # Completely suppress stderr
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)

    try:
        ray.init(
            logging_level=logging.CRITICAL,
            log_to_driver=True,
            include_dashboard=False,
        )

        # Filter stdout during simulation
        filtered_stdout = OutputFilter(sys.__stdout__)
        old_stdout = sys.stdout
        sys.stdout = filtered_stdout

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.num_clients,
            config=ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
            ray_init_args={"logging_level": logging.CRITICAL, "log_to_driver": True},
        )

        sys.stdout = old_stdout
    finally:
        # Restore stderr
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        os.close(devnull_fd)

    # Save results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

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
