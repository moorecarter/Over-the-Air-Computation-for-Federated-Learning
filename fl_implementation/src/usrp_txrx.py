from .usrp_x310 import USRP_X310
import numpy as np
import threading
import time
from typing import Dict, List, Optional
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
        u.set_clk(clk_source={"internal" if u.usrp.addr == server_addr else "external"}, time_source={"internal" if u.usrp.addr == server_addr else "external"})

    # Wait for PPS edge, set time to 0 on all at the next PPS
    time.sleep(1.1) 
    for u in all_usrps:
        u.usrp.set_time_next_pps(uhd.types.TimeSpec(0.0))

    # Wait for that PPS
    time.sleep(1.1)
    # USRPS have T=0 

def usrp_channel_estimation(
    num_clients: int,
    server_round: int,
    server_usrp_addr: str,
    client_usrp_addr: List[str],
):
    print("Starting USRP channel estimation...")
    csi = np.ones(num_clients, dtype=np.complex64)

    server_usrp = _get_usrp(server_usrp_addr)
    if not server_usrp.rx_channels:
        server_usrp.set_rx(freq=2.45e9, samprate=1e6, gain=25, channel=0, antenna="TX/RX", lo_offset=1e6)
    rx_symbols = None

    def rx_thread_fn():
        nonlocal rx_symbols
        rx_symbols = server_usrp.rx_pilot(num_samps=256 + 2000)

    for client_idx in range(num_clients):
        client_usrp = _get_usrp(client_usrp_addr[client_idx])
        if not client_usrp.tx_channels:
            client_usrp.set_tx(freq=2.45e9, samprate=1e6, gain=25, channel=0, antenna="TX/RX", lo_offset=1e6)
        rx_thread = threading.Thread(target=rx_thread_fn)
        rx_thread.start()
        time.sleep(0.05)
        client_usrp.tx_pilot(amplitude=1.0, pilot=KNOWN_PILOT)
        time.sleep(0.05)
        rx_thread.join()

        if rx_symbols is not None and len(rx_symbols) >= N_ZC:
            cross_corr = np.correlate(rx_symbols, KNOWN_PILOT, mode='valid')
            peak_idx = np.argmax(np.abs(cross_corr))
            peak_val = cross_corr[peak_idx]

            # CSI is the complex channel coefficient (magnitude + phase)
            csi[client_idx] = peak_val

            print(f"Client {client_idx} - CSI: |h|={np.abs(peak_val):.4f}, phase={np.angle(peak_val):.2f} rad")
        else:
            print(f"Error: Did not receive enough samples from client {client_idx}")

    return csi

def create_usrp_channel_estimate_callback(server_usrp_addr: str, client_usrp_addr: List[str]):
    """Strategy only calls (num_clients, server_round); this binds addresses."""
    return lambda num_clients, server_round: usrp_channel_estimation(
        num_clients, server_round, server_usrp_addr, client_usrp_addr
    )

def usrp_transmit_and_receive(
    num_clients: int,
    server_round: int,
    server_usrp_addr: str,
    client_usrp_addr: List[str],
    client_signals: List[np.ndarray],
    channel_state: List[np.complex64],
    weights: Optional[List[float]] = None
):

    server_usrp = _get_usrp(server_usrp_addr)
    if not server_usrp.rx_channels:
        server_usrp.set_rx(freq=2.45e9, samprate=1e6, gain=25, channel=0, antenna="TX/RX", lo_offset=1e6)
    client_usrps = [_get_usrp(addr) for addr in client_usrp_addr]
    for client_usrp in client_usrps:
        if not client_usrp.tx_channels:
            client_usrp.set_tx(freq=2.45e9, samprate=1e6, gain=25, channel=0, antenna="TX/RX", lo_offset=1e6)

    if weights is None:
        weights = np.ones(num_clients) / num_clients
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
    
    encoded_signals = []
    for i, sig in enumerate(client_signals):
        weighted_grads = sig.flatten() * weights[i]
        waveform = USRP_X310.grad_to_wave(grads=weighted_grads, amplitude=0.2, csi=channel_state[i])
        encoded_signals.append(waveform)

    start_time = server_usrp.usrp.get_time_now() + uhd.libpyuhd.types.TimeSpec(0.5)
    rx_symbols = None
    def rx_thread_fn():
        nonlocal rx_symbols
        rx_symbols = server_usrp.rx_signal(num_samps=len(encoded_signals[0]) + 2000, start_time=start_time)
    def tx_thread_fn(client_idx: int):
        client_usrps[client_idx].tx_signal(waveform=encoded_signals[client_idx], repeat=False, start_time=start_time)
    
    print("Starting USRP transmit and receive...")

    #Start RX and TX threads
    rx_thread = threading.Thread(target=rx_thread_fn)
    tx_threads = [threading.Thread(target=tx_thread_fn, args=(i,)) for i in range(num_clients)]
    rx_thread.start()
    for t in tx_threads:
        t.start()
    for t in tx_threads:
        t.join()
    rx_thread.join()
    rx_grads = USRP_X310.wave_to_grad(wave=rx_symbols, amplitude=0.2)
    rx_grads = rx_grads[:client_signals[0].flatten().shape[0]].reshape(client_signals[0].shape)
    return rx_grads

def create_usrp_transmit_and_receive_callback(server_usrp_addr: str, client_usrp_addr: List[str]):
    def callback(client_deltas, weights, param_idx, server_round, channel_state, **kwargs):
        # 1. Derive num_clients from the incoming data
        num_clients = len(client_deltas)
        
        # 2. Map the new caller variables to your USRP function's expected inputs.
        # Assuming 'client_deltas' maps to what you previously called 'client_signals'
        client_signals = client_deltas 
        
        # 3. Call your underlying USRP function
        # Note: If your usrp_transmit_and_receive needs channel_state, weights, 
        # or param_idx, you'll need to pass them in here too!
        return usrp_transmit_and_receive(
            num_clients,
            server_round,
            server_usrp_addr,
            client_usrp_addr,
            client_signals,
            channel_state,
            weights
        )

    return callback