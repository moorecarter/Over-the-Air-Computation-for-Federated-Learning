#!/usr/bin/env python3
"""
USRP Test 5: CSI (Channel State Information) Estimation
Test channel estimation between two USRPs using pilot signals
This is critical for OTA computation to compensate for channel effects
"""

import sys
import os
sys.path.append('../src')

from Utils.usrp_x310 import USRP_X310
from Utils.csi_estimate import csi_estimate
import numpy as np
import argparse
import time

def test_csi_estimation(rx_addr, tx_addr):
    """Test CSI estimation between two USRPs"""
    print("=== USRP TEST 5: CSI Estimation ===")
    print(f"RX USRP: {rx_addr}")
    print(f"TX USRP: {tx_addr}\n")
    
    # Configuration
    FC = 2.45e9
    FS = 1e6
    RX_GAIN = 30
    TX_GAIN = 10  # Low for wired test
    
    print("Configuration:")
    print(f"  Frequency: {FC/1e9:.3f} GHz")
    print(f"  Sample Rate: {FS/1e6:.1f} MHz")
    print(f"  RX Gain: {RX_GAIN} dB")
    print(f"  TX Gain: {TX_GAIN} dB\n")
    
    # Initialize USRPs
    print("Initializing USRPs...")
    usrp_rx = USRP_X310(rx_addr)
    usrp_tx = USRP_X310(tx_addr)
    
    # Setup RX
    usrp_rx.set_rx(
        freq=FC,
        samprate=FS,
        gain=RX_GAIN,
        channel=0,
        antenna="RX2"
    )
    
    # Setup TX
    usrp_tx.set_tx(
        freq=FC,
        samprate=FS,
        gain=TX_GAIN,
        channel=0,
        antenna="TX/RX"
    )
    
    print("USRPs configured\n")
    
    # Perform multiple CSI estimates to check stability
    print("Performing CSI estimation (5 measurements)...")
    csi_measurements = []
    
    for i in range(5):
        print(f"\nMeasurement {i+1}:")
        
        # Estimate CSI
        h = csi_estimate(device_tx=usrp_tx, device_rx=usrp_rx)
        csi_measurements.append(h)
        
        # Display results
        magnitude = np.abs(h)
        phase_deg = np.angle(h) * 180 / np.pi
        
        print(f"  CSI = {h:.6f}")
        print(f"  Magnitude: {magnitude:.6f}")
        print(f"  Phase: {phase_deg:.2f}°")
        
        # Wait between measurements
        if i < 4:
            time.sleep(1)
    
    # Analyze CSI stability
    print("\n=== CSI Stability Analysis ===")
    
    csi_array = np.array(csi_measurements)
    mag_array = np.abs(csi_array)
    phase_array = np.angle(csi_array)
    
    print(f"Magnitude statistics:")
    print(f"  Mean: {np.mean(mag_array):.6f}")
    print(f"  Std Dev: {np.std(mag_array):.6f}")
    print(f"  Max: {np.max(mag_array):.6f}")
    print(f"  Min: {np.min(mag_array):.6f}")
    
    print(f"\nPhase statistics (degrees):")
    print(f"  Mean: {np.mean(phase_array)*180/np.pi:.2f}°")
    print(f"  Std Dev: {np.std(phase_array)*180/np.pi:.2f}°")
    
    # Check if CSI is stable
    mag_variation = np.std(mag_array) / np.mean(mag_array) * 100
    print(f"\nMagnitude variation: {mag_variation:.1f}%")
    
    if mag_variation < 5:
        print("✅ Excellent channel stability!")
    elif mag_variation < 10:
        print("✅ Good channel stability")
    elif mag_variation < 20:
        print("⚠️  Moderate channel stability")
    else:
        print("⚠️  High channel variation - check connections")
    
    # Test using CSI for compensation
    print("\n=== Testing CSI Compensation ===")
    
    # Send test gradients
    test_grads = np.array([1.0, -0.5, 0.8, -0.3, 0.6])
    print(f"Test gradients: {test_grads}")
    
    # Encode and transmit
    waveform, scale = usrp_tx.grad_to_wave(
        grads=test_grads,
        amplitude=0.2,
        sps=100
    )
    
    print("Transmitting test gradients...")
    
    # Receive
    rx_signal = usrp_rx.rx_signal(len(waveform))
    
    # Decode without CSI compensation
    recovered_no_csi = USRP_X310.wave_to_grad(
        wave=rx_signal,
        amplitude=0.2,
        sps=100,
        csi=1.0,  # No compensation
        scale=scale
    )[:len(test_grads)]
    
    # Decode with CSI compensation
    recovered_with_csi = USRP_X310.wave_to_grad(
        wave=rx_signal,
        amplitude=0.2,
        sps=100,
        csi=csi_measurements[-1],  # Use latest CSI
        scale=scale
    )[:len(test_grads)]
    
    print(f"\nRecovered without CSI: {recovered_no_csi}")
    print(f"Recovered with CSI:    {recovered_with_csi}")
    
    # Calculate errors
    error_no_csi = np.mean((recovered_no_csi - test_grads)**2)
    error_with_csi = np.mean((recovered_with_csi - test_grads)**2)
    
    print(f"\nMSE without CSI: {error_no_csi:.6f}")
    print(f"MSE with CSI:    {error_with_csi:.6f}")
    
    if error_with_csi < error_no_csi:
        improvement = (error_no_csi - error_with_csi) / error_no_csi * 100
        print(f"\n✅ CSI compensation improved accuracy by {improvement:.1f}%!")
    
    print("\n✅ CSI estimation test complete!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CSI estimation")
    parser.add_argument("--rx-addr", default="192.168.10.2",
                        help="RX USRP IP address")
    parser.add_argument("--tx-addr", default="192.168.10.3",
                        help="TX USRP IP address")
    args = parser.parse_args()
    
    test_csi_estimation(args.rx_addr, args.tx_addr)