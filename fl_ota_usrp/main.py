#!/usr/bin/env python3
"""
Federated Learning with Over-the-Air Computation using USRP X310
Main entry point for both server and client modes
"""

import argparse
import sys
import torch
import torch.nn as nn
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from fl_usrp_server import FLUSRPServer
from fl_usrp_client import FLUSRPClient


# Simple CNN model for demonstration
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data_for_client(client_id: int, num_clients: int, batch_size: int = 32):
    """
    Load and partition data for a specific client
    For now, uses synthetic data - replace with actual dataset
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # Generate synthetic data (replace with real data loading)
    # Simulating MNIST-like data: 28x28 grayscale images
    num_samples_per_client = 100
    x = torch.randn(num_samples_per_client, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples_per_client,))
    
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def run_server(args):
    """Run the FL server"""
    print("=== Starting FL OTA Server ===")
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'fc': 2.45e9,
            'fs': 1e6,
            'rx_gain': 30,
            'rx_ant': 'RX2',
            'rx_chan': 0,
            'num_clients': args.num_clients,
            'num_rounds': args.num_rounds,
            'lr': 0.01,
            'amplitude': 0.2,
            'sps': 100
        }
    
    # Create model
    model = SimpleCNN(num_classes=10)
    
    # Create and run server
    server = FLUSRPServer(
        usrp_addr=args.usrp_addr,
        model=model,
        config=config
    )
    
    server.run_fl_training()


def run_client(args):
    """Run the FL client"""
    print(f"=== Starting FL OTA Client {args.client_id} ===")
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'fc': 2.45e9,
            'fs': 1e6,
            'tx_gain': 15,
            'tx_ant': 'TX/RX',
            'tx_chan': 0,
            'local_epochs': 5,
            'lr': 0.01
        }
    
    # Create model
    model = SimpleCNN(num_classes=10)
    
    # Load data for this client
    train_loader = load_data_for_client(
        client_id=args.client_id,
        num_clients=3,  # Total number of clients
        batch_size=32
    )
    
    # Create and run client
    client = FLUSRPClient(
        client_id=args.client_id,
        usrp_addr=args.usrp_addr,
        server_addr=args.server_addr,
        model=model,
        train_loader=train_loader,
        config=config
    )
    
    client.run_fl_client()


def main():
    parser = argparse.ArgumentParser(
        description='Federated Learning with OTA Computation using USRP'
    )
    
    # Mode selection
    parser.add_argument('--mode', choices=['server', 'client'], required=True,
                        help='Run as server or client')
    
    # USRP configuration
    parser.add_argument('--usrp-addr', required=True,
                        help='IP address of USRP device')
    
    # Client-specific arguments
    parser.add_argument('--client-id', type=int,
                        help='Client ID (required for client mode)')
    parser.add_argument('--server-addr', default='localhost',
                        help='Server IP address (for client mode)')
    
    # Server-specific arguments
    parser.add_argument('--num-clients', type=int, default=3,
                        help='Number of clients (for server mode)')
    parser.add_argument('--num-rounds', type=int, default=10,
                        help='Number of FL rounds (for server mode)')
    
    # Optional configuration file
    parser.add_argument('--config', type=str,
                        help='Path to JSON configuration file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'client' and args.client_id is None:
        parser.error('--client-id is required for client mode')
    
    # Run appropriate mode
    if args.mode == 'server':
        run_server(args)
    else:
        run_client(args)


if __name__ == '__main__':
    main()