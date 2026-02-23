import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import socket
import pickle
from typing import Dict, Tuple
from Utils import USRP_X310


class FLUSRPClient:
    def __init__(self, client_id: int, usrp_addr: str, server_addr: str, 
                 model: nn.Module, train_loader: DataLoader, config: dict):
        """
        Federated Learning Client with USRP for OTA transmission
        
        Args:
            client_id: Unique client identifier
            usrp_addr: IP address of client USRP
            server_addr: IP address of server
            model: PyTorch model to train
            train_loader: DataLoader with local training data
            config: Configuration dict with FL and USRP parameters
        """
        self.client_id = client_id
        self.usrp_addr = usrp_addr
        self.server_addr = server_addr
        self.model = model
        self.train_loader = train_loader
        self.config = config
        
        # USRP parameters
        self.fc = config.get('fc', 2.45e9)  # Center frequency
        self.fs = config.get('fs', 1e6)     # Sample rate
        self.tx_gain = config.get('tx_gain', 15)
        self.tx_ant = config.get('tx_ant', 'TX/RX')
        self.tx_chan = config.get('tx_chan', 0)
        
        # FL parameters
        self.local_epochs = config.get('local_epochs', 5)
        self.lr = config.get('lr', 0.01)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize USRP
        self.usrp = USRP_X310(usrp_addr)
        self.usrp.set_tx(
            freq=self.fc,
            samprate=self.fs,
            gain=self.tx_gain,
            channel=self.tx_chan,
            antenna=self.tx_ant
        )
        
        # Connect to server
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.connect((server_addr, 5555))
        print(f"Client {client_id} connected to server at {server_addr}")
        
        # Store gradients
        self.gradients = None
        
    def receive_model(self):
        """Receive model from server"""
        # Receive model size
        size_bytes = self.control_socket.recv(8)
        model_size = int.from_bytes(size_bytes, 'big')
        
        # Receive model
        model_bytes = b''
        while len(model_bytes) < model_size:
            chunk = self.control_socket.recv(min(4096, model_size - len(model_bytes)))
            model_bytes += chunk
            
        # Load model
        model_state = pickle.loads(model_bytes)
        self.model.load_state_dict(model_state)
        print(f"Client {self.client_id}: Received model from server")
        
    def local_training(self):
        """Perform local training on client data"""
        print(f"Client {self.client_id}: Starting local training...")
        
        # Store initial model state
        initial_state = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Setup optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.local_epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.train_loader)
            print(f"Client {self.client_id}: Epoch {epoch+1}/{self.local_epochs}, Loss: {avg_loss:.4f}")
            
        # Calculate gradients (difference between new and old model)
        self.gradients = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.gradients[name] = (initial_state[name] - param).cpu().numpy()
                
        print(f"Client {self.client_id}: Local training complete")
        
    def participate_in_csi_estimation(self):
        """Participate in channel estimation"""
        # Wait for server's PILOT request
        msg = self.control_socket.recv(1024)
        if msg == b'PILOT':
            # Send client info
            client_info = {
                'addr': self.usrp_addr,
                'tx_gain': self.tx_gain,
                'tx_chan': self.tx_chan,
                'tx_ant': self.tx_ant
            }
            self.control_socket.send(pickle.dumps(client_info))
            
            # Wait for CSI completion
            msg = self.control_socket.recv(1024)
            if msg == b'CSI_DONE':
                print(f"Client {self.client_id}: CSI estimation complete")
                
    def transmit_gradients_ota(self):
        """Transmit gradients via OTA"""
        # Wait for server's PREPARE_GRADIENTS
        msg = self.control_socket.recv(1024)
        if msg != b'PREPARE_GRADIENTS':
            return
            
        # Flatten gradients
        grad_list = []
        param_shapes = []
        param_names = []
        for name, grad in self.gradients.items():
            grad_list.append(grad.flatten())
            param_shapes.append(grad.shape)
            param_names.append(name)
        gradients_flat = np.concatenate(grad_list)
        
        # Send READY
        self.control_socket.send(b'READY')
        
        # Send metadata if requested
        msg = self.control_socket.recv(1024)
        if msg == b'GET_METADATA':
            metadata = {
                'total_params': len(gradients_flat),
                'param_shapes': param_shapes,
                'param_names': param_names
            }
            self.control_socket.send(pickle.dumps(metadata))
            
            # Wait for next message
            msg = self.control_socket.recv(1024)
            
        # Receive transmission parameters
        if msg == b'TRANSMIT':
            tx_params = pickle.loads(self.control_socket.recv(4096))
            sync_time = tx_params['sync_time']
            amplitude = tx_params['amplitude']
            sps = tx_params['sps']
            
            # Encode gradients to waveform
            waveform, scale = self.usrp.grad_to_wave(
                grads=gradients_flat,
                amplitude=amplitude,
                sps=sps
            )
            
            # Wait until sync time
            wait_time = sync_time - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
                
            # Transmit!
            print(f"Client {self.client_id}: Transmitting gradients OTA at {time.time()}")
            self.usrp.tx_signal(waveform=waveform, repeat=False)
            
            # Confirm transmission
            self.control_socket.send(b'TX_COMPLETE')
            print(f"Client {self.client_id}: OTA transmission complete")
            
    def run_fl_client(self):
        """Main client loop"""
        print(f"Client {self.client_id}: Starting FL client")
        
        # Initial CSI estimation
        self.participate_in_csi_estimation()
        
        round_idx = 0
        while True:
            round_idx += 1
            print(f"\nClient {self.client_id}: Round {round_idx}")
            
            # 1. Receive global model
            self.receive_model()
            
            # 2. Local training
            self.local_training()
            
            # 3. Send training complete signal
            self.control_socket.send(b'TRAINING_DONE')
            
            # 4. Transmit gradients via OTA
            self.transmit_gradients_ota()
            
            # Check if training is complete
            try:
                self.control_socket.settimeout(0.1)
                msg = self.control_socket.recv(1024, socket.MSG_DONTWAIT)
                if msg == b'TRAINING_COMPLETE':
                    print(f"Client {self.client_id}: Training complete!")
                    break
                elif msg == b'PILOT':
                    # Re-estimate CSI
                    self.control_socket.settimeout(None)
                    self.participate_in_csi_estimation()
            except:
                pass
            finally:
                self.control_socket.settimeout(None)
                
        # Clean up
        self.control_socket.close()
        print(f"Client {self.client_id}: Disconnected")