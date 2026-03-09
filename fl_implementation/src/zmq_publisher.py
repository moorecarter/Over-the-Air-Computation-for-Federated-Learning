"""
ZeroMQ PUB socket for streaming FL telemetry to dashboard subscribers.

Usage:
    pub = ZMQPublisher(port=5555)
    pub.send_config({...})       # once at startup
    pub.send_status(...)         # during rounds
    pub.send_round(...)          # end of each round
    pub.close()
"""

import json
import zmq
import numpy as np


class ZMQPublisher:
    def __init__(self, port: int = 5555):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.bind(f"tcp://0.0.0.0:{port}")
        print(f"[ZMQ] Publishing on tcp://0.0.0.0:{port}")

    def send_config(
        self,
        total_params: int,
        batch_size: int,
        local_epochs: int,
        num_clients: int,
        model: str,
        dataset: str,
        freq_hz: float = 2.41e9,
        sample_rate: float = 1e6,
    ):
        payload = json.dumps({
            "total_params": total_params,
            "batch_size": batch_size,
            "local_epochs": local_epochs,
            "num_clients": num_clients,
            "model": model,
            "dataset": dataset,
            "freq_hz": freq_hz,
            "sample_rate": sample_rate,
        })
        self._socket.send_multipart([b"config", payload.encode()])

    def send_status(self, state: str, round: int, client_id: int = None):
        msg = {"state": state, "round": round}
        if client_id is not None:
            msg["client_id"] = client_id
        self._socket.send_multipart([b"status", json.dumps(msg).encode()])

    def send_round(
        self,
        round: int,
        accuracy: float,
        loss: float,
        tx_buffer: np.ndarray = None,
        rx_buffer: np.ndarray = None,
    ):
        metadata = json.dumps({
            "round": round,
            "accuracy": accuracy,
            "loss": loss,
            "tx_len": len(tx_buffer) if tx_buffer is not None else 0,
            "rx_len": len(rx_buffer) if rx_buffer is not None else 0,
        })
        frames = [b"round", metadata.encode()]
        frames.append(tx_buffer.tobytes() if tx_buffer is not None else b"")
        frames.append(rx_buffer.tobytes() if rx_buffer is not None else b"")
        self._socket.send_multipart(frames)

    def close(self):
        self._socket.close()
        self._ctx.term()
