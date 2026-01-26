import uhd 
import numpy as np
import usrp_x310 
import time
import threading

def csi_estimate(device_tx: usrp_x310, device_rx: usrp_x310):
    AMPLITUDE = 0.2
    SPS = 100

    # Prepare pilot waveform first (so RX knows num_samps)
    pilot_sequence = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32)
    known_pilot, _ = device_tx.grad_to_wave(grads=pilot_sequence, amplitude=AMPLITUDE, sps=SPS)
    num_samps = len(known_pilot)

    # Start RX in a thread BEFORE TX
    rx_symbols = None
    def rx_thread_fn():
        nonlocal rx_symbols
        rx_symbols = device_rx.rx_pilot(num_samps=num_samps, sps=SPS)

    rx_thread = threading.Thread(target=rx_thread_fn)
    rx_thread.start()
    time.sleep(0.05)  # let RX settle

    # TX burst
    device_tx.tx_signal(waveform=known_pilot, repeat=False)

    # Wait for RX to finish
    rx_thread.join()

    # Compute channel
    csi = np.mean(rx_symbols / (known_pilot / np.max(np.abs(known_pilot))))

    return csi