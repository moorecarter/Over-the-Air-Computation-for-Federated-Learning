"""
ZeroMQ PUB socket for streaming FL telemetry to dashboard subscribers.

All sends run on a dedicated background thread so they never block
the USRP / Flower thread.

Usage:
    pub = ZMQPublisher(port=5555)
    pub.send_config({...})       # once at startup
    pub.send_status(...)         # during rounds
    pub.send_metrics(...)        # end of each round
    pub.close()
"""

import json
import threading
import queue
import zmq
import numpy as np


class ZMQPublisher:
    def __init__(self, port: int = 5555):
        self._queue = queue.Queue()
        self._port = port
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUB)
        socket.bind(f"tcp://0.0.0.0:{self._port}")
        print(f"[ZMQ] Publishing on tcp://0.0.0.0:{self._port}")

        while True:
            frames = self._queue.get()
            if frames is None:
                break
            socket.send_multipart(frames)

        socket.close()
        ctx.term()

    def send_config(
        self,
        total_params: int,
        batch_size: int,
        local_epochs: int,
        num_clients: int,
        num_rounds: int,
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
            "num_rounds": num_rounds,
            "model": model,
            "dataset": dataset,
            "freq_hz": freq_hz,
            "sample_rate": sample_rate,
        })
        self._queue.put([b"config", payload.encode()])

    def send_status(self, state: str, round: int):
        msg = {"state": state, "round": round}
        self._queue.put([b"status", json.dumps(msg).encode()])

    def send_metrics(
        self,
        accuracy: float,
        loss: float,
        csi_per_client: list = None,
        rx_buffer: list = None,
        time_offsets_ns: list = None,
        snr_db: list = None,
    ):
        # Normalize optional RX buffer: send magnitudes as floats
        rx_buffer_mag = []
        if isinstance(rx_buffer, np.ndarray):
            rx_buffer_mag = [float(np.abs(x)) for x in rx_buffer]
        elif rx_buffer is not None:
            for x in rx_buffer:
                rx_buffer_mag.append(float(np.abs(complex(x))))

        payload = json.dumps({
            "accuracy": accuracy,
            "loss": loss,
            "csi_per_client": csi_per_client or [],
            "rx_buffer_mag": rx_buffer_mag,
            "time_offsets_ns": time_offsets_ns or [],
            "snr_db": snr_db or [],
        })
        self._queue.put([b"metrics", payload.encode()])

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=2)
