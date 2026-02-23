"""
Over-The-Air (OTA) Aggregation Strategy for Flower.

Simulates the wireless channel effects that occur during OTA computation:
- Channel noise (AWGN)
- Fading effects
- Synchronization errors
- Power imbalance between clients

This allows testing federated learning algorithms under realistic
OTA conditions before deploying to actual USRP hardware.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class OTAChannelModel:
    """
    Simulates wireless channel effects for OTA computation.

    In real OTA aggregation:
    - Clients transmit simultaneously over wireless channel
    - Signals superimpose in the air (natural summation)
    - Receiver gets sum + channel effects

    This class models those effects for simulation.
    """

    def __init__(
        self,
        snr_db: float = 20.0,
        fading_type: str = "none",  # "none", "rayleigh", "rician"
        rician_k: float = 3.0,  # K-factor for Rician fading
        sync_error_std: float = 0.0,  # Phase sync error (radians)
        power_imbalance_db: float = 0.0,  # Max power difference between clients
        seed: Optional[int] = None,
    ):
        """
        Args:
            snr_db: Signal-to-noise ratio in dB
            fading_type: Type of fading model
            rician_k: K-factor for Rician fading (higher = more LOS component)
            sync_error_std: Standard deviation of phase synchronization error
            power_imbalance_db: Maximum power imbalance between clients (dB)
            seed: Random seed for reproducibility
        """
        self.snr_db = snr_db
        self.fading_type = fading_type
        self.rician_k = rician_k
        self.sync_error_std = sync_error_std
        self.power_imbalance_db = power_imbalance_db

        self.rng = np.random.default_rng(seed)

    def compute_noise_power(self, signal_power: float) -> float:
        """Compute noise power from SNR."""
        snr_linear = 10 ** (self.snr_db / 10)
        return signal_power / snr_linear

    def apply_fading(self, num_clients: int) -> np.ndarray:
        """
        Generate fading coefficients for each client.

        Returns:
            Array of complex fading coefficients, one per client
        """
        if self.fading_type == "none":
            return np.ones(num_clients, dtype=complex)

        elif self.fading_type == "rayleigh":
            # Rayleigh fading: magnitude follows Rayleigh distribution
            real = self.rng.standard_normal(num_clients)
            imag = self.rng.standard_normal(num_clients)
            h = (real + 1j * imag) / np.sqrt(2)
            return h

        elif self.fading_type == "rician":
            # Rician fading: LOS component + scattered components
            k = self.rician_k
            los = np.sqrt(k / (k + 1))  # LOS component
            scatter_std = np.sqrt(1 / (2 * (k + 1)))

            real = los + self.rng.standard_normal(num_clients) * scatter_std
            imag = self.rng.standard_normal(num_clients) * scatter_std
            h = real + 1j * imag
            return h

        else:
            raise ValueError(f"Unknown fading type: {self.fading_type}")

    def apply_sync_error(self, num_clients: int) -> np.ndarray:
        """
        Generate phase errors due to imperfect synchronization.

        Returns:
            Array of phase offsets (radians), one per client
        """
        if self.sync_error_std == 0:
            return np.zeros(num_clients)

        return self.rng.normal(0, self.sync_error_std, num_clients)

    def apply_power_imbalance(self, num_clients: int) -> np.ndarray:
        """
        Generate power scaling factors due to imbalanced transmit power.

        Returns:
            Array of power scaling factors, one per client
        """
        if self.power_imbalance_db == 0:
            return np.ones(num_clients)

        # Random power offset for each client (in dB)
        power_offsets_db = self.rng.uniform(
            -self.power_imbalance_db / 2,
            self.power_imbalance_db / 2,
            num_clients,
        )
        return 10 ** (power_offsets_db / 20)  # Convert to linear scale

    def aggregate_with_channel(
        self,
        client_updates: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Simulate OTA aggregation with channel effects.

        In ideal OTA: received = sum(client_signals)
        With channel: received = sum(h_i * g_i * exp(j*phi_i) * client_i) + noise

        Args:
            client_updates: List of parameter arrays from clients
            weights: Optional weights for each client (e.g., by dataset size)

        Returns:
            Aggregated parameters with channel effects applied
        """
        num_clients = len(client_updates)

        if weights is None:
            weights = np.ones(num_clients) / num_clients
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize

        # Get channel effects
        fading_coeffs = self.apply_fading(num_clients)
        phase_errors = self.apply_sync_error(num_clients)
        power_scales = self.apply_power_imbalance(num_clients)

        # Combined channel coefficient (magnitude only for real-valued params)
        # In real OTA, would need proper pre-coding to align phases
        channel_gains = np.abs(fading_coeffs) * power_scales * np.cos(phase_errors)

        # Effective weights after channel effects
        effective_weights = weights * channel_gains
        effective_weights = effective_weights / effective_weights.sum()  # Renormalize

        # Weighted sum of updates
        aggregated = np.zeros_like(client_updates[0])
        for i, update in enumerate(client_updates):
            aggregated += effective_weights[i] * update

        # Add AWGN noise
        signal_power = np.mean(aggregated ** 2)
        noise_power = self.compute_noise_power(signal_power)
        noise = self.rng.normal(0, np.sqrt(noise_power), aggregated.shape)
        aggregated += noise

        return aggregated


class FedAvgOTA(FedAvg):
    """
    Federated Averaging with simulated OTA channel effects.

    Extends the standard FedAvg strategy to apply wireless channel
    impairments during aggregation, simulating over-the-air computation.

    For USRP integration:
    - Clients send DELTAS (weight changes), not full weights
    - Server aggregates deltas via OTA (or USRP callback)
    - Aggregated delta is applied to global model
    """

    def __init__(
        self,
        *,
        # OTA channel parameters
        snr_db: float = 20.0,
        fading_type: str = "none",
        rician_k: float = 3.0,
        sync_error_std: float = 0.0,
        power_imbalance_db: float = 0.0,
        channel_seed: Optional[int] = None,
        # USRP integration
        usrp_callback: Optional[callable] = None,  # Custom transmission function
        # Standard FedAvg parameters
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
    ):
        """
        Args:
            snr_db: Channel SNR in dB (higher = less noise)
            fading_type: "none", "rayleigh", or "rician"
            rician_k: K-factor for Rician fading
            sync_error_std: Phase synchronization error std (radians)
            power_imbalance_db: Max power imbalance between clients
            channel_seed: Random seed for channel simulation
            usrp_callback: Optional callback for USRP transmission.
                           Signature: callback(client_deltas, weights) -> aggregated_delta
                           If None, uses simulated OTA channel.

            (Other args are standard FedAvg parameters)
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        self.channel = OTAChannelModel(
            snr_db=snr_db,
            fading_type=fading_type,
            rician_k=rician_k,
            sync_error_std=sync_error_std,
            power_imbalance_db=power_imbalance_db,
            seed=channel_seed,
        )

        # USRP callback for real OTA transmission
        self.usrp_callback = usrp_callback

        # Store global parameters (updated each round)
        self._global_parameters: Optional[List[np.ndarray]] = None
        if initial_parameters is not None:
            self._global_parameters = parameters_to_ndarrays(initial_parameters)

        # Store channel config for reference
        self.channel_config_str = f"OTA: SNR={snr_db}dB, fading={fading_type}"

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates with OTA channel effects.

        Clients send DELTAS (weight changes). Server aggregates via OTA
        and applies to global model.
        """
        if not results:
            return None, {}

        # Check for failures
        if not self.accept_failures and failures:
            return None, {}

        # Extract deltas and sample counts from clients
        client_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics)
            for _, fit_res in results
        ]

        # Check if clients sent deltas or full weights
        is_delta = client_results[0][2].get("is_delta", True)

        # Get total examples for weighting
        num_examples_total = sum([n for _, n, _ in client_results])
        weights = [n / num_examples_total for _, n, _ in client_results]

        # Aggregate each parameter array separately
        aggregated_ndarrays = []
        num_params = len(client_results[0][0])

        for param_idx in range(num_params):
            # Collect this parameter's delta from all clients
            client_deltas = [result[0][param_idx] for result in client_results]

            # === USRP TRANSMISSION POINT ===
            # This is where deltas would be transmitted over the air
            # client_deltas: list of numpy arrays, one per client
            # weights: importance of each client (by sample count)

            if self.usrp_callback is not None:
                # Use real USRP for OTA aggregation
                aggregated_delta = self.usrp_callback(
                    client_deltas=client_deltas,
                    weights=weights,
                    param_idx=param_idx,
                    server_round=server_round,
                )
            else:
                # Use simulated OTA channel
                aggregated_delta = self.channel.aggregate_with_channel(
                    client_deltas, weights
                )

            if is_delta:
                # Clients sent deltas - apply to global parameters
                if self._global_parameters is not None:
                    new_param = self._global_parameters[param_idx] + aggregated_delta
                else:
                    # First round - just use the delta (shouldn't happen normally)
                    new_param = aggregated_delta
                aggregated_ndarrays.append(new_param)
            else:
                # Clients sent full weights - use directly (standard FedAvg)
                aggregated_ndarrays.append(aggregated_delta)

        # Update stored global parameters
        self._global_parameters = aggregated_ndarrays

        # Convert back to Parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate metrics (standard)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

    def get_client_deltas_for_usrp(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[List[List[np.ndarray]], List[float]]:
        """
        Extract client deltas in a format ready for USRP transmission.

        Returns:
            Tuple of (client_deltas, weights) where:
            - client_deltas[client_idx][param_idx] = numpy array of deltas
            - weights[client_idx] = weight for this client
        """
        client_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        num_examples_total = sum([n for _, n in client_results])
        weights = [n / num_examples_total for _, n in client_results]

        client_deltas = [deltas for deltas, _ in client_results]

        return client_deltas, weights


def create_ota_strategy(
    channel_config: str = "ideal",
    **kwargs,
) -> FedAvgOTA:
    """
    Create an OTA strategy with preset channel configurations.

    Args:
        channel_config: Preset configuration name
            - "ideal": No channel effects (SNR=inf)
            - "good": High SNR, no fading (indoor, close range)
            - "moderate": Medium SNR, Rician fading (indoor, some obstacles)
            - "challenging": Low SNR, Rayleigh fading (outdoor, mobile)
            - "harsh": Very low SNR, severe fading (worst case)

    Returns:
        Configured FedAvgOTA strategy
    """
    presets = {
        "ideal": dict(
            snr_db=100.0,
            fading_type="none",
            sync_error_std=0.0,
            power_imbalance_db=0.0,
        ),
        "good": dict(
            snr_db=30.0,
            fading_type="none",
            sync_error_std=0.01,
            power_imbalance_db=1.0,
        ),
        "moderate": dict(
            snr_db=20.0,
            fading_type="rician",
            rician_k=5.0,
            sync_error_std=0.05,
            power_imbalance_db=3.0,
        ),
        "challenging": dict(
            snr_db=15.0,
            fading_type="rayleigh",
            sync_error_std=0.1,
            power_imbalance_db=6.0,
        ),
        "harsh": dict(
            snr_db=10.0,
            fading_type="rayleigh",
            sync_error_std=0.2,
            power_imbalance_db=10.0,
        ),
    }

    if channel_config not in presets:
        raise ValueError(f"Unknown preset: {channel_config}. Available: {list(presets.keys())}")

    config = presets[channel_config]
    config.update(kwargs)

    return FedAvgOTA(**config)


if __name__ == "__main__":
    # Test channel model
    print("Testing OTA Channel Model")
    print("=" * 60)

    # Create some fake client updates
    np.random.seed(42)
    num_clients = 5
    param_shape = (100,)

    # Simulated parameter updates from clients
    client_updates = [np.random.randn(*param_shape) for _ in range(num_clients)]

    # Test different channel conditions
    for config_name in ["ideal", "good", "moderate", "challenging", "harsh"]:
        channel = OTAChannelModel(
            **{k: v for k, v in {
                "ideal": dict(snr_db=100.0, fading_type="none"),
                "good": dict(snr_db=30.0, fading_type="none"),
                "moderate": dict(snr_db=20.0, fading_type="rician"),
                "challenging": dict(snr_db=15.0, fading_type="rayleigh"),
                "harsh": dict(snr_db=10.0, fading_type="rayleigh"),
            }[config_name].items()},
            seed=42,
        )

        # Ideal aggregation (simple average)
        ideal_result = np.mean(client_updates, axis=0)

        # OTA aggregation
        ota_result = channel.aggregate_with_channel(client_updates)

        # Compute MSE between ideal and OTA
        mse = np.mean((ideal_result - ota_result) ** 2)
        print(f"{config_name:12s}: MSE = {mse:.6f}")

    print("\nChannel model tests complete!")
