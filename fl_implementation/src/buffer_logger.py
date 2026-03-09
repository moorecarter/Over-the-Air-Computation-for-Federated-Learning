"""
Logs the raw USRP RX buffer to disk for analysis.
Saves the complex64 samples as a .npy file on the first round.
"""

import numpy as np
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "outputs" / "buffer_logs"
_logged_rounds = set()


def log_rx_buffer(rx_symbols: np.ndarray, server_round: int, only_first: bool = True):
    """
    Save the raw RX buffer to a .npy file.

    Args:
        rx_symbols: Raw complex64 samples from the USRP RX
        server_round: Current FL round number
        only_first: If True, only log round 1
    """
    if only_first and server_round != 1:
        return
    if server_round in _logged_rounds:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"rx_buffer_round_{server_round}.npy"
    np.save(path, rx_symbols)
    _logged_rounds.add(server_round)
    print(f"  [BufferLogger] Saved RX buffer ({len(rx_symbols)} samples) to {path}")
