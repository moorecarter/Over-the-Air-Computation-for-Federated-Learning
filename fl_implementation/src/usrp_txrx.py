from .usrp_x310 import USRP_X310
import numpy as np
import threading
import time
from typing import List, Optional
import math
import uhd.libpyuhd.types
def _generate_zadoff_chu(N: int, u: int, q: int = 0) -> np.ndarray:
            
    if(math.gcd(N, u) != 1):
        raise Exception("Please choose N and u such that they are coprime.")
    
    n = np.arange(N)
    cf = N % 2
    arg = -1j * (np.pi * u * n * (n + cf + 2 * q)) / N
    x = np.exp(arg).astype(np.complex64)
    return x

KNOWN_PILOT = _generate_zadoff_chu(N=256, u=1, q=0)
KNOWN_PILOT_WAVEFORM = USRP_X310.grad_to_wave(grads=KNOWN_PILOT, amplitude=1.0)
N_ZC = len(KNOWN_PILOT) 

_usrp_cache = {}
def _get_usrp(ip_addr: str) -> USRP_X310:
    if ip_addr not in _usrp_cache:
        _usrp_cache[ip_addr] = USRP_X310(ip_addr=ip_addr)
    return _usrp_cache[ip_addr]
    
#NEED TO LOOK THIS IMPL OVER 
def init_all_usrps(server_addr: str, client_addrs: List[str]):
    """Call once at startup before any rounds."""
    all_usrps = [_get_usrp(server_addr)] + [_get_usrp(addr) for addr in client_addrs]
    # ============================    CHANGE TO EXTERNAL WHEN OCTOCLOCK ADDED   ============================
    for u in all_usrps:
        u.set_clk(clk_source="external", time_source="external")
        while not u.usrp.get_mboard_sensor("ref_locked").to_bool():
            time.sleep(0.1)

    # Wait for PPS edge, set time to 0 on all at the next PPS
    time.sleep(1.1) 
    for u in all_usrps:
        u.usrp.set_time_next_pps(uhd.libpyuhd.types.time_spec(0.0))

    # Wait for that PPS
    time.sleep(1.1)
    # USRPS have T=0

    # Pre-configure RX on server, TX on clients so radios are ready before round 1
    server = _get_usrp(server_addr)
    server.set_rx(freq=2.41e9, samprate=1e6, gain=20, channel=0, antenna="TX/RX", lo_offset=1e6)
    for addr in client_addrs:
        _get_usrp(addr).set_tx(freq=2.41e9, samprate=1e6, gain=25, channel=0, antenna="TX/RX", lo_offset=1e6)
    time.sleep(0.5)  # Let radios settle

def usrp_channel_estimation(
    num_clients: int,
    server_round: int,
    server_usrp_addr: str,
    client_usrp_addr: List[str],
):
    # print("Starting USRP channel estimation...")
    csi = np.ones(num_clients, dtype=np.complex64)

    server_usrp = _get_usrp(server_usrp_addr)
    if not server_usrp.rx_channels:
        server_usrp.set_rx(freq=2.41e9, samprate=1e6, gain=20, channel=0, antenna="TX/RX", lo_offset=1e6)
    rx_symbols = None

    SAMPLE_RATE = 1e6
    peak_indices = np.zeros(num_clients, dtype=int)
    snr_per_client = np.zeros(num_clients)

    for client_idx in range(num_clients):
        rx_symbols = None
        client_usrp = _get_usrp(client_usrp_addr[client_idx])
        if not client_usrp.tx_channels:
            client_usrp.set_tx(freq=2.41e9, samprate=1e6, gain=25, channel=0, antenna="TX/RX", lo_offset=1e6)

        # Use timed commands so RX and TX start at exactly the same moment
        start_time = server_usrp.usrp.get_time_now() + uhd.libpyuhd.types.time_spec(0.05)

        def rx_thread_fn():
            nonlocal rx_symbols
            rx_symbols = server_usrp.rx_signal(num_samps=256 + 2000, start_time=start_time)

        def tx_thread_fn():
            client_usrp.tx_signal(waveform=KNOWN_PILOT_WAVEFORM, repeat=False, start_time=start_time)

        rx_thread = threading.Thread(target=rx_thread_fn)
        tx_thread = threading.Thread(target=tx_thread_fn)
        rx_thread.start()
        tx_thread.start()
        tx_thread.join()
        rx_thread.join()

        if rx_symbols is not None and len(rx_symbols) >= N_ZC:
            cross_corr = np.correlate(rx_symbols, KNOWN_PILOT_WAVEFORM, mode='valid')
            peak_idx = np.argmax(np.abs(cross_corr))
            peak_val = cross_corr[peak_idx]
            csi[client_idx] = peak_val / N_ZC
            peak_indices[client_idx] = peak_idx

            # SNR estimate: signal = pilot region, noise = samples before pilot
            signal_region = rx_symbols[peak_idx:peak_idx + N_ZC]
            noise_region = rx_symbols[:max(peak_idx, 1)]
            signal_power = np.mean(np.abs(signal_region) ** 2)
            noise_power = np.mean(np.abs(noise_region) ** 2)
            if noise_power > 0:
                snr_per_client[client_idx] = 10 * np.log10(signal_power / noise_power)

    # Time offsets relative to first client (in nanoseconds)
    time_offsets_ns = ((peak_indices - peak_indices[0]) / SAMPLE_RATE * 1e9).tolist()

    return {
        "csi": csi,
        "time_offsets_ns": time_offsets_ns,
        "snr_db": snr_per_client.tolist(),
    }

def create_usrp_channel_estimate_callback(server_usrp_addr: str, client_usrp_addr: List[str]):
    """Strategy only calls (num_clients, server_round); this binds addresses."""
    return lambda num_clients, server_round: usrp_channel_estimation(
        num_clients, server_round, server_usrp_addr, client_usrp_addr
    )

GUARD_LEN = 64  # Guard interval between pilot and data


def usrp_transmit_and_receive(
    num_clients: int,
    server_round: int,
    server_usrp_addr: str,
    client_usrp_addr: List[str],
    client_signals: List[np.ndarray],
    channel_state: dict,
    weights: Optional[List[float]] = None,
):
    server_usrp = _get_usrp(server_usrp_addr)
    if not server_usrp.rx_channels:
        server_usrp.set_rx(freq=2.41e9, samprate=1e6, gain=20, channel=0, antenna="TX/RX", lo_offset=1e6)
    client_usrps = [_get_usrp(addr) for addr in client_usrp_addr]
    for client_usrp in client_usrps:
        if not client_usrp.tx_channels:
            client_usrp.set_tx(freq=2.41e9, samprate=1e6, gain=25, channel=0, antenna="TX/RX", lo_offset=1e6)

    if weights is None:
        weights = np.ones(num_clients) / num_clients
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()

    csi_array = channel_state["csi"] if channel_state is not None else None
    signal_len = len(client_signals[0].flatten())

    # Compute adaptive amplitude: find worst-case peak across all clients
    max_precoded = 0.0
    for i, sig in enumerate(client_signals):
        weighted = np.abs(sig.flatten() * weights[i])
        csi_mag = max(np.abs(csi_array[i]), 0.01) if csi_array is not None else 1.0
        peak = np.max(weighted) / csi_mag
        if peak > max_precoded:
            max_precoded = peak

    TARGET_PEAK = 0.8
    amplitude = TARGET_PEAK / max_precoded if max_precoded > 0 else TARGET_PEAK
    # print(f"  [Round {server_round}] Adaptive amplitude: {amplitude:.6f} (max_precoded: {max_precoded:.6f})")

    # Encode: [pilot | guard | data] per client — pilot enables alignment at RX
    encoded_signals = []
    for i, sig in enumerate(client_signals):
        weighted_grads = sig.flatten() * weights[i]
        csi = csi_array[i] if csi_array is not None else 1.0
        if np.abs(csi) < 0.01:
            print(f"  [Warning] Client {i} CSI magnitude too low ({np.abs(csi):.6f}), clamping")
            csi = 0.01 * np.exp(1j * np.angle(csi))
        data_waveform = USRP_X310.grad_to_wave(grads=weighted_grads, amplitude=amplitude, csi=csi)
        tx_waveform = np.concatenate([
            KNOWN_PILOT_WAVEFORM,
            np.zeros(GUARD_LEN, dtype=np.complex64),
            data_waveform,
        ])
        encoded_signals.append(tx_waveform)

    tx_len = len(encoded_signals[0])

    # RX captures extra margin to account for timing offset
    rx_num_samps = tx_len + 4000
    start_time = server_usrp.usrp.get_time_now() + uhd.libpyuhd.types.time_spec(0.5)
    rx_symbols = None
    rx_ready = threading.Event()

    def rx_thread_fn():
        nonlocal rx_symbols
        rx_ready.set()
        rx_symbols = server_usrp.rx_signal(num_samps=rx_num_samps, start_time=start_time)

    def tx_thread_fn(client_idx: int):
        rx_ready.wait()
        client_usrps[client_idx].tx_signal(waveform=encoded_signals[client_idx], repeat=False, start_time=start_time)

    rx_thread = threading.Thread(target=rx_thread_fn)
    tx_threads = [threading.Thread(target=tx_thread_fn, args=(i,)) for i in range(num_clients)]
    rx_thread.start()
    for t in tx_threads:
        t.start()
    for t in tx_threads:
        t.join()
    rx_thread.join()

    if rx_symbols is None:
        raise RuntimeError(f"[Round {server_round}] RX failed: no samples received from USRP")

    # Find signal start via pilot cross-correlation
    cross_corr = np.correlate(rx_symbols, KNOWN_PILOT_WAVEFORM, mode='valid')
    peak_idx = np.argmax(np.abs(cross_corr))

    # Data starts after pilot + guard
    data_start = peak_idx + N_ZC + GUARD_LEN
    data_end = data_start + signal_len

    if data_end > len(rx_symbols):
        data_end = len(rx_symbols)

    rx_data_for_grad = rx_symbols[data_start:data_end]
    rx_grads = USRP_X310.wave_to_grad(waveform=rx_data_for_grad, amplitude=amplitude)

    # Pad if we got fewer samples than expected
    if len(rx_grads) < signal_len:
        rx_grads = np.concatenate([rx_grads, np.zeros(signal_len - len(rx_grads))])

    return rx_grads[:signal_len], rx_symbols


def create_usrp_transmit_and_receive_callback(server_usrp_addr: str, client_usrp_addr: List[str]):
    """Create a bulk OTA callback for the strategy.

    Signature matches what FedAvgOTA.aggregate_fit expects:
        callback(client_deltas, weights, server_round, channel_state) -> aggregated flat array
    """
    def callback(client_deltas, weights, server_round, channel_state=None, **kwargs):
        num_clients = len(client_deltas)
        try:
            aggregated_flat, rx_data = usrp_transmit_and_receive(
                num_clients,
                server_round,
                server_usrp_addr,
                client_usrp_addr,
                client_deltas,
                channel_state,
                weights,
            )
            # Make RX samples available for ZMQ publishing
            max_points = 2048
            if rx_data is not None and len(rx_data) > 0:
                step = max(1, len(rx_data) // max_points)
                callback.rx_buffer = rx_data[::step]
            return aggregated_flat
        except Exception as e:
            print(f"[Round {server_round}] USRP callback error: {e}")
            import traceback
            traceback.print_exc()
            raise

    # Initialize attribute so callers can safely read it before first round
    callback.rx_buffer = np.array([], dtype=np.complex64)
    return callback