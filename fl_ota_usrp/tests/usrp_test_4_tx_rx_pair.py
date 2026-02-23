#!/usr/bin/env python3
"""
USRP Test 4: TX-RX Pair (Wired Connection)
Test transmission and reception between two USRPs
Run this on the RX machine, it will coordinate with TX
"""

import uhd
import numpy as np
import argparse
import time
import socket
import threading

def rx_thread_func(usrp_addr, received_signal, event):
    """RX thread to receive signal"""
    print(f"RX: Initializing USRP at {usrp_addr}...")
    
    # Configuration
    freq = 2.45e9
    rate = 1e6
    gain = 30
    
    # Create USRP
    usrp = uhd.usrp.MultiUSRP(f"addr={usrp_addr}")
    
    # Configure RX
    usrp.set_rx_rate(rate)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq), 0)
    usrp.set_rx_gain(gain, 0)
    usrp.set_rx_antenna("RX2", 0)
    
    # Setup streamer
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    rx_streamer = usrp.get_rx_stream(st_args)
    
    # Number of samples to receive
    num_samps = int(rate * 0.5)  # 0.5 seconds
    received_signal.resize(num_samps, refcheck=False)
    
    print("RX: Ready to receive")
    event.set()  # Signal that RX is ready
    
    # Start streaming
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samps
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)
    
    # Receive samples
    metadata = uhd.types.RXMetadata()
    samples_received = 0
    
    while samples_received < num_samps:
        num_rx = rx_streamer.recv(
            received_signal[samples_received:], metadata
        )
        samples_received += num_rx
    
    print(f"RX: Received {samples_received} samples")

def test_tx_rx_pair(rx_addr, tx_addr):
    """Test TX-RX communication between USRPs"""
    print("=== USRP TEST 4: TX-RX Pair Test ===")
    print(f"RX USRP: {rx_addr}")
    print(f"TX USRP: {tx_addr}\n")
    
    # Create signal to transmit
    print("Creating test signal...")
    rate = 1e6
    duration = 0.5
    num_samples = int(rate * duration)
    t = np.arange(num_samples) / rate
    
    # Create chirp signal (frequency sweep)
    f_start = 50e3
    f_end = 200e3
    chirp = np.exp(1j * 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration)))
    tx_signal = 0.5 * chirp.astype(np.complex64)
    
    print(f"  Signal: Chirp {f_start/1e3:.0f}-{f_end/1e3:.0f} kHz")
    print(f"  Duration: {duration} seconds")
    print(f"  Samples: {num_samples}\n")
    
    # Start RX thread
    received_signal = np.zeros(num_samples, dtype=np.complex64)
    rx_ready = threading.Event()
    rx_thread = threading.Thread(
        target=rx_thread_func,
        args=(rx_addr, received_signal, rx_ready)
    )
    rx_thread.start()
    
    # Wait for RX to be ready
    rx_ready.wait()
    time.sleep(0.5)  # Give RX time to start streaming
    
    # Transmit signal
    print(f"TX: Initializing USRP at {tx_addr}...")
    usrp_tx = uhd.usrp.MultiUSRP(f"addr={tx_addr}")
    
    # Configure TX
    usrp_tx.set_tx_rate(rate)
    usrp_tx.set_tx_freq(uhd.libpyuhd.types.tune_request(2.45e9), 0)
    usrp_tx.set_tx_gain(10, 0)  # Low gain for wired connection
    usrp_tx.set_tx_antenna("TX/RX", 0)
    
    # Setup TX streamer
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    tx_streamer = usrp_tx.get_tx_stream(st_args)
    
    # Transmit
    print("TX: Transmitting signal...")
    metadata = uhd.types.TXMetadata()
    metadata.start_of_burst = True
    metadata.end_of_burst = True
    metadata.has_time_spec = False
    
    tx_streamer.send(tx_signal, metadata)
    
    # Wait for RX to complete
    rx_thread.join()
    
    # Analyze received signal
    print("\n=== Analysis ===")
    
    # Calculate correlation
    correlation = np.correlate(received_signal, tx_signal, mode='valid')
    max_corr = np.max(np.abs(correlation))
    
    # Calculate power
    tx_power = np.mean(np.abs(tx_signal)**2)
    rx_power = np.mean(np.abs(received_signal)**2)
    
    # Calculate SNR (rough estimate)
    signal_power = np.max(np.abs(received_signal)**2)
    noise_floor = np.mean(np.abs(received_signal[:1000])**2)  # First 1000 samples
    snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-12))
    
    print(f"TX Power: {10*np.log10(tx_power):.1f} dB")
    print(f"RX Power: {10*np.log10(rx_power + 1e-12):.1f} dB")
    print(f"Path Loss: {10*np.log10(tx_power/(rx_power + 1e-12)):.1f} dB")
    print(f"SNR: {snr_db:.1f} dB")
    print(f"Correlation: {max_corr:.6f}")
    
    # Check success
    if rx_power > 1e-6:
        print("\n✅ TEST PASSED: Signal successfully transmitted and received!")
        
        if snr_db > 30:
            print("Excellent signal quality!")
        elif snr_db > 20:
            print("Good signal quality")
        elif snr_db > 10:
            print("Moderate signal quality")
        else:
            print("Low signal quality - consider adjusting gains")
    else:
        print("\n❌ TEST FAILED: No signal received")
        print("Check:")
        print("- Cable connection between USRPs")
        print("- Attenuators (30dB recommended for wired)")
        print("- TX and RX frequencies match")
    
    return rx_power > 1e-6

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TX-RX pair")
    parser.add_argument("--rx-addr", default="192.168.10.2",
                        help="RX USRP IP address")
    parser.add_argument("--tx-addr", default="192.168.10.3",
                        help="TX USRP IP address")
    args = parser.parse_args()
    
    test_tx_rx_pair(args.rx_addr, args.tx_addr)