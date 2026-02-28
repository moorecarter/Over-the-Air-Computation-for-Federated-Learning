from .usrp_x310 import USRP_X310
import numpy as np
import threading
import time
from typing import List

# Known pilot symbols (same as USRP_X310.tx_pilot) for FFT reference
KNOWN_PILOT = np.array([1, -1, 1, -1, 1, -1, 1, -1] + [0] * 800, dtype=np.complex64)


def usrp_channel_estimation(
    num_clients: int,
    server_round: int,
    server_usrp_addr: str,
    client_usrp_addr: List[str],
):
    # 1. Run pilot phase on host USRP (e.g. receive known pilots from each client)
    fading_coeffs = np.zeros(num_clients)
    phase_errors = np.zeros(num_clients)
    power_scales = np.zeros(num_clients)

    server_usrp = USRP_X310(ip_addr=server_usrp_addr)
    server_usrp.set_rx(freq=2.4e9, samprate=1e6, gain=25, channel=0, antenna="TX/RX")
    rx_symbols = None

    def rx_thread_fn():
        nonlocal rx_symbols
        rx_symbols = server_usrp.rx_pilot(num_samps=1000, sps=100)

    for client_idx in range(num_clients):
        client_usrp = USRP_X310(ip_addr=client_usrp_addr[client_idx])
        rx_thread = threading.Thread(target=rx_thread_fn)
        rx_thread.start()
        time.sleep(0.05)  # let RX settle
        pilot = client_usrp.tx_pilot(amplitude=1.0, sps=100)
        time.sleep(0.05)
        rx_thread.join()
        # FFT: use known pilot for reference (same length as rx for correlation)
        x = np.zeros(256, dtype=np.complex64)
        x[: len(KNOWN_PILOT)] = KNOWN_PILOT
        y = np.zeros(256, dtype=np.complex64)
        n_rx = min(len(rx_symbols) if rx_symbols is not None else 0, 256)
        if rx_symbols is not None:
            y[:n_rx] = rx_symbols[:n_rx]
        y_f = np.fft.fft(y, 256)
        x_f = np.fft.fft(x, 256)
        h_est_conj_f = y_f * np.conjugate(x_f)
        cir_conj = np.fft.ifft(h_est_conj_f, 256)
        # Get phase and magnitude at peak tap (scalar per client)
        peak_idx = np.argmax(np.abs(cir_conj))
        fading_coeffs[client_idx] = np.abs(cir_conj[peak_idx])
        phase_errors[client_idx] = np.angle(cir_conj[peak_idx])
        power_scales[client_idx] = np.abs(cir_conj[peak_idx]) / (np.abs(KNOWN_PILOT).mean() + 1e-12)

    return {"fading_coeffs": fading_coeffs, "phase_errors": phase_errors, "power_scales": power_scales}


def create_usrp_channel_estimate_callback(server_usrp_addr: str, client_usrp_addr: List[str]):
    """Strategy only calls (num_clients, server_round); this binds addresses."""
    return lambda num_clients, server_round: usrp_channel_estimation(
        num_clients, server_round, server_usrp_addr, client_usrp_addr
    )
