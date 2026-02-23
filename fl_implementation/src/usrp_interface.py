"""
USRP Interface for Over-The-Air (OTA) Federated Learning.

This module provides the interface between the FL server and USRP X310
hardware for real over-the-air computation.

Usage:
    1. Implement the USRPTransmitter class with your USRP setup
    2. Pass usrp_callback to FedAvgOTA strategy
    3. Deltas from each client will be transmitted over the air

Example:
    from src.usrp_interface import create_usrp_callback

    # Create callback for USRP transmission
    usrp_callback = create_usrp_callback(
        usrp_ip="192.168.10.2",
        sample_rate=1e6,
        center_freq=2.4e9,
    )

    # Use in strategy
    strategy = FedAvgOTA(
        usrp_callback=usrp_callback,
        ...
    )
"""

import numpy as np
from typing import List, Optional, Callable
from abc import ABC, abstractmethod


class USRPTransmitter(ABC):
    """
    Abstract base class for USRP transmission.

    Implement this class with your specific USRP setup.
    """

    @abstractmethod
    def transmit_and_receive(
        self,
        client_signals: List[np.ndarray],
        weights: List[float],
    ) -> np.ndarray:
        """
        Transmit client signals over the air and receive aggregated result.

        In real OTA computation:
        1. Each client transmits its signal simultaneously
        2. Signals superimpose in the wireless channel
        3. Server receives the sum (natural aggregation)

        Args:
            client_signals: List of signals to transmit (one per client)
            weights: Importance weight for each client

        Returns:
            Aggregated signal received over the air
        """
        pass


class SimulatedUSRP(USRPTransmitter):
    """
    Simulated USRP for testing without hardware.

    Simulates ideal OTA aggregation (simple weighted average).
    """

    def __init__(self, add_noise: bool = False, snr_db: float = 30.0):
        self.add_noise = add_noise
        self.snr_db = snr_db

    def transmit_and_receive(
        self,
        client_signals: List[np.ndarray],
        weights: List[float],
    ) -> np.ndarray:
        """Simulate OTA transmission with optional noise."""
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted sum (OTA aggregation)
        aggregated = np.zeros_like(client_signals[0])
        for signal, weight in zip(client_signals, weights):
            aggregated += weight * signal

        # Optionally add noise
        if self.add_noise:
            signal_power = np.mean(aggregated ** 2)
            snr_linear = 10 ** (self.snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = np.random.normal(0, np.sqrt(noise_power), aggregated.shape)
            aggregated += noise

        return aggregated


class USRPX310(USRPTransmitter):
    """
    Real USRP X310 implementation.

    TODO: Implement with UHD Python API or GNU Radio.

    Example UHD setup:
        import uhd

        usrp = uhd.usrp.MultiUSRP("addr=192.168.10.2")
        usrp.set_rx_rate(sample_rate)
        usrp.set_tx_rate(sample_rate)
        usrp.set_rx_freq(center_freq)
        usrp.set_tx_freq(center_freq)
    """

    def __init__(
        self,
        usrp_ip: str = "192.168.10.2",
        sample_rate: float = 1e6,
        center_freq: float = 2.4e9,
        tx_gain: float = 30.0,
        rx_gain: float = 30.0,
    ):
        """
        Initialize USRP X310 connection.

        Args:
            usrp_ip: IP address of USRP X310
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz
            tx_gain: Transmit gain in dB
            rx_gain: Receive gain in dB
        """
        self.usrp_ip = usrp_ip
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain

        # TODO: Initialize UHD connection
        # self.usrp = uhd.usrp.MultiUSRP(f"addr={usrp_ip}")
        # self.usrp.set_rx_rate(sample_rate)
        # self.usrp.set_tx_rate(sample_rate)
        # self.usrp.set_rx_freq(center_freq)
        # self.usrp.set_tx_freq(center_freq)
        # self.usrp.set_tx_gain(tx_gain)
        # self.usrp.set_rx_gain(rx_gain)

        print(f"USRP X310 interface initialized (IP: {usrp_ip})")
        print(f"  Sample rate: {sample_rate/1e6:.1f} MHz")
        print(f"  Center freq: {center_freq/1e9:.2f} GHz")

    def transmit_and_receive(
        self,
        client_signals: List[np.ndarray],
        weights: List[float],
    ) -> np.ndarray:
        """
        Transmit client signals over USRP and receive aggregated result.

        TODO: Implement actual USRP transmission.

        For now, this is a placeholder that simulates the transmission.
        Replace with actual UHD calls for real hardware.
        """
        # TODO: Implement actual USRP transmission
        # Steps:
        # 1. Modulate each client's delta values to baseband signal
        # 2. Coordinate transmission timing across clients
        # 3. Transmit signals simultaneously
        # 4. Receive and demodulate aggregated signal
        # 5. Return demodulated values

        # Placeholder: simulate ideal aggregation
        weights = np.array(weights)
        weights = weights / weights.sum()

        aggregated = np.zeros_like(client_signals[0])
        for signal, weight in zip(client_signals, weights):
            aggregated += weight * signal

        return aggregated

    def modulate(self, values: np.ndarray) -> np.ndarray:
        """
        Modulate float values to baseband signal for transmission.

        TODO: Implement modulation scheme (e.g., OFDM, QAM).
        """
        # Placeholder: direct mapping (not suitable for real transmission)
        return values.astype(np.complex64)

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """
        Demodulate received signal back to float values.

        TODO: Implement demodulation matching the modulation scheme.
        """
        # Placeholder: direct mapping
        return signal.real.astype(np.float32)


def create_usrp_callback(
    transmitter: Optional[USRPTransmitter] = None,
    use_simulation: bool = True,
    **kwargs,
) -> Callable:
    """
    Create a callback function for USRP transmission.

    This callback is passed to FedAvgOTA and called for each parameter
    layer during aggregation.

    Args:
        transmitter: USRPTransmitter instance (or None to create one)
        use_simulation: If True, use SimulatedUSRP; else use USRPX310
        **kwargs: Arguments passed to transmitter constructor

    Returns:
        Callback function with signature:
        callback(client_deltas, weights, param_idx, server_round) -> aggregated_delta
    """
    if transmitter is None:
        if use_simulation:
            transmitter = SimulatedUSRP(**kwargs)
        else:
            transmitter = USRPX310(**kwargs)

    def usrp_callback(
        client_deltas: List[np.ndarray],
        weights: List[float],
        param_idx: int,
        server_round: int,
    ) -> np.ndarray:
        """
        Callback for USRP OTA transmission.

        Args:
            client_deltas: Delta arrays from each client for this parameter
            weights: Importance weight for each client
            param_idx: Index of the parameter being aggregated
            server_round: Current FL round number

        Returns:
            Aggregated delta after OTA transmission
        """
        # Transmit and receive via USRP
        aggregated = transmitter.transmit_and_receive(client_deltas, weights)

        return aggregated

    return usrp_callback


# Convenience function to get deltas in a format ready for external processing
def prepare_deltas_for_transmission(
    client_deltas: List[List[np.ndarray]],
    weights: List[float],
) -> dict:
    """
    Prepare client deltas in a format suitable for external transmission systems.

    This can be used to export deltas for processing by external OTA systems
    (e.g., GNU Radio flowgraphs, custom USRP scripts).

    Args:
        client_deltas: client_deltas[client_idx][param_idx] = numpy array
        weights: Weight for each client

    Returns:
        Dictionary with transmission-ready data
    """
    num_clients = len(client_deltas)
    num_params = len(client_deltas[0])

    # Flatten all parameters into single vectors per client
    client_vectors = []
    for client_idx in range(num_clients):
        client_vector = np.concatenate([
            delta.flatten() for delta in client_deltas[client_idx]
        ])
        client_vectors.append(client_vector)

    # Stack into matrix: (num_clients, total_params)
    delta_matrix = np.stack(client_vectors)

    return {
        "delta_matrix": delta_matrix,  # Shape: (num_clients, total_params)
        "weights": np.array(weights),
        "num_clients": num_clients,
        "num_params": num_params,
        "total_values": delta_matrix.shape[1],
        "param_shapes": [delta.shape for delta in client_deltas[0]],
    }


def reconstruct_from_aggregated(
    aggregated_vector: np.ndarray,
    param_shapes: List[tuple],
) -> List[np.ndarray]:
    """
    Reconstruct parameter arrays from aggregated flat vector.

    Args:
        aggregated_vector: Flat vector of aggregated values
        param_shapes: Original shapes of each parameter array

    Returns:
        List of parameter arrays with original shapes
    """
    params = []
    offset = 0
    for shape in param_shapes:
        size = np.prod(shape)
        param = aggregated_vector[offset:offset + size].reshape(shape)
        params.append(param)
        offset += size
    return params


if __name__ == "__main__":
    # Test the USRP interface
    print("Testing USRP Interface")
    print("=" * 60)

    # Create simulated USRP
    usrp = SimulatedUSRP(add_noise=True, snr_db=20.0)

    # Simulate 3 clients with some delta values
    np.random.seed(42)
    client_deltas = [
        np.random.randn(100) * 0.01 for _ in range(3)
    ]
    weights = [0.33, 0.33, 0.34]

    # Transmit and receive
    aggregated = usrp.transmit_and_receive(client_deltas, weights)

    # Compare with ideal aggregation
    ideal = sum(w * d for w, d in zip(weights, client_deltas))
    mse = np.mean((aggregated - ideal) ** 2)

    print(f"Simulated OTA transmission:")
    print(f"  Input shape: {client_deltas[0].shape}")
    print(f"  Output shape: {aggregated.shape}")
    print(f"  MSE vs ideal: {mse:.6f}")

    # Test callback creation
    print("\nTesting callback creation:")
    callback = create_usrp_callback(use_simulation=True, add_noise=False)
    result = callback(client_deltas, weights, param_idx=0, server_round=1)
    print(f"  Callback result shape: {result.shape}")
    print(f"  First 5 values: {result[:5]}")

    print("\nUSRP interface tests complete!")
