import numpy as np
import torch
import torch.nn as nn
import time
import threading
import json
import socket
import pickle
from typing import Dict, List, Tuple
from Utils import USRP_X310, csi_estimate


class FLUSRPServer:
    def __init__(self, usrp_addr: str, model: nn.Module, config: dict):
        """
        Federated Learning Server with USRP for OTA aggregation
        
        Args:
            usrp_addr: IP address of server USRP
            model: PyTorch model to train
            config: Configuration dict with FL and USRP parameters
        """
        self.usrp_addr = usrp_addr
        self.model = model
        self.config = config
        
        # USRP parameters
        self.fc = config.get('fc', 2.45e9)  # Center frequency
        self.fs = config.get('fs', 1e6)     # Sample rate
        self.rx_gain = config.get('rx_gain', 30)
        self.rx_ant = config.get('rx_ant', 'RX2')
        self.rx_chan = config.get('rx_chan', 0)
        
        # FL parameters
        self.num_clients = config.get('num_clients', 3)
        self.num_rounds = config.get('num_rounds', 10)
        
        # Initialize USRP
        self.usrp = USRP_X310(usrp_addr)
        self.usrp.set_rx(
            freq=self.fc,
            samprate=self.fs,
            gain=self.rx_gain,
            channel=self.rx_chan,
            antenna=self.rx_ant
        )
        
        # Store CSI for each client
        self.client_csi = {}
        
        # Communication socket for control messages
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_socket.bind(('0.0.0.0', 5555))
        self.control_socket.listen(self.num_clients)
        
        self.client_connections = []
        
    def wait_for_clients(self):
        """Wait for all clients to connect"""
        print(f"Waiting for {self.num_clients} clients to connect...")
        for i in range(self.num_clients):
            conn, addr = self.control_socket.accept()
            self.client_connections.append(conn)
            print(f"Client {i} connected from {addr}")
        print("All clients connected!")
        
    def broadcast_model(self):
        """Send current model to all clients"""
        model_state = self.model.state_dict()
        model_bytes = pickle.dumps(model_state)
        
        for i, conn in enumerate(self.client_connections):
            # Send model size first
            conn.send(len(model_bytes).to_bytes(8, 'big'))
            # Send model
            conn.send(model_bytes)
            print(f"Sent model to client {i}")
            
    def estimate_channels(self):
        """Estimate CSI for each client"""
        print("Estimating channels for all clients...")
        
        for i, conn in enumerate(self.client_connections):
            # Tell client to transmit pilot
            conn.send(b'PILOT')
            
            # Wait for client to setup TX
            ack = conn.recv(1024)
            
            # Create temporary TX USRP object for CSI estimation
            client_info = pickle.loads(ack)
            client_usrp = USRP_X310(client_info['addr'])
            client_usrp.set_tx(
                freq=self.fc,
                samprate=self.fs,
                gain=client_info['tx_gain'],
                channel=client_info['tx_chan'],
                antenna=client_info['tx_ant']
            )
            
            # Estimate CSI
            csi = csi_estimate(device_tx=client_usrp, device_rx=self.usrp)
            self.client_csi[i] = csi
            print(f"CSI for client {i}: magnitude={np.abs(csi):.4f}, phase={np.angle(csi):.4f}")
            
            # Tell client CSI is done
            conn.send(b'CSI_DONE')
            
    def receive_ota_gradients(self) -> Dict:
        """
        Coordinate OTA gradient reception from all clients
        
        Returns:
            Aggregated gradients as a dict matching model.state_dict() structure
        """
        print("Starting OTA gradient reception...")
        
        # Tell all clients to prepare gradients
        for conn in self.client_connections:
            conn.send(b'PREPARE_GRADIENTS')
        
        # Wait for all clients to be ready
        ready_count = 0
        for conn in self.client_connections:
            msg = conn.recv(1024)
            if msg == b'READY':
                ready_count += 1
        
        print(f"{ready_count}/{self.num_clients} clients ready")
        
        # Get gradient metadata from first client
        self.client_connections[0].send(b'GET_METADATA')
        metadata = pickle.loads(self.client_connections[0].recv(4096))
        total_params = metadata['total_params']
        param_shapes = metadata['param_shapes']
        param_names = metadata['param_names']
        
        # Parameters for gradient encoding
        amplitude = self.config.get('amplitude', 0.2)
        sps = self.config.get('sps', 100)  # samples per symbol
        
        # Calculate expected number of samples
        num_samps = total_params * sps
        
        # Tell all clients to transmit simultaneously
        sync_time = time.time() + 2.0  # 2 seconds from now
        for conn in self.client_connections:
            conn.send(b'TRANSMIT')
            conn.send(pickle.dumps({
                'sync_time': sync_time,
                'amplitude': amplitude,
                'sps': sps
            }))
        
        # Start receiving before transmission time
        time.sleep(1.5)
        print("Receiving OTA aggregated signal...")
        rx_signal = self.usrp.rx_signal(num_samps)
        
        # Wait for clients to confirm transmission
        for conn in self.client_connections:
            msg = conn.recv(1024)
            
        # Decode aggregated gradients
        # Since signals add in the air, we need to account for that
        # The received signal is the sum of all client signals
        
        # For now, simplified decoding (will need channel compensation)
        aggregated_grads_flat = USRP_X310.wave_to_grad(
            wave=rx_signal,
            amplitude=amplitude * self.num_clients,  # Scale for multiple clients
            sps=sps,
            csi=1.0,  # Simplified - should use combined CSI
            scale=1.0
        )
        
        # Reshape gradients back to model structure
        aggregated_grads = {}
        idx = 0
        for name, shape in zip(param_names, param_shapes):
            param_size = np.prod(shape)
            param_flat = aggregated_grads_flat[idx:idx+param_size]
            aggregated_grads[name] = param_flat.reshape(shape)
            idx += param_size
            
        print("OTA aggregation complete!")
        return aggregated_grads
    
    def update_model(self, aggregated_grads: Dict):
        """Update model with aggregated gradients"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_grads:
                    # Average the gradients
                    grad_avg = torch.tensor(aggregated_grads[name] / self.num_clients)
                    # Update model using gradient descent
                    param.data -= self.config.get('lr', 0.01) * grad_avg
                    
    def run_fl_training(self):
        """Main FL training loop"""
        print(f"Starting FL training for {self.num_rounds} rounds")
        
        # Wait for clients
        self.wait_for_clients()
        
        # Initial CSI estimation
        self.estimate_channels()
        
        # FL rounds
        for round_idx in range(self.num_rounds):
            print(f"\n=== Round {round_idx + 1}/{self.num_rounds} ===")
            
            # 1. Broadcast current model
            self.broadcast_model()
            
            # 2. Wait for clients to train
            print("Waiting for clients to complete local training...")
            for conn in self.client_connections:
                msg = conn.recv(1024)  # Wait for TRAINING_DONE
                
            # 3. Receive aggregated gradients via OTA
            aggregated_grads = self.receive_ota_gradients()
            
            # 4. Update global model
            self.update_model(aggregated_grads)
            
            # 5. Optional: Evaluate model
            # TODO: Add evaluation on test set
            
            # 6. Re-estimate channels periodically
            if (round_idx + 1) % 5 == 0:
                print("Re-estimating channels...")
                self.estimate_channels()
                
        print("\nFL training complete!")
        
        # Close connections
        for conn in self.client_connections:
            conn.send(b'TRAINING_COMPLETE')
            conn.close()
        self.control_socket.close()