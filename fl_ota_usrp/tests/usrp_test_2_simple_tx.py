#!/usr/bin/env python3
"""
USRP Test 2: Simple Transmission
Test transmitting a simple signal from one USRP
"""

import uhd
import numpy as np
import argparse
import time

def test_simple_tx(addr):
    """Test basic signal transmission"""
    print(f"=== USRP TEST 2: Simple Transmission ===")
    print(f"Using USRP at {addr}\n")
    
    # Configuration
    freq = 2.45e9
    rate = 1e6
    gain = 10  # Start with low gain
    channels = [0]
    
    print("Configuration:")
    print(f"  Frequency: {freq/1e9:.3f} GHz")
    print(f"  Sample Rate: {rate/1e6:.1f} MHz")
    print(f"  TX Gain: {gain} dB")
    print(f"  Antenna: TX/RX\n")
    
    # Create USRP
    print("Initializing USRP...")
    usrp = uhd.usrp.MultiUSRP(f"addr={addr}")
    
    # Configure TX
    usrp.set_tx_rate(rate)
    for chan in channels:
        usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(freq), chan)
        usrp.set_tx_gain(gain, chan)
        usrp.set_tx_antenna("TX/RX", chan)
    
    # Create test signal (simple sine wave)
    print("Creating test signal...")
    duration = 1.0  # 1 second
    num_samples = int(rate * duration)
    t = np.arange(num_samples) / rate
    f_tone = 100e3  # 100 kHz tone
    
    # Complex sinusoid
    signal = 0.7 * np.exp(2j * np.pi * f_tone * t).astype(np.complex64)
    print(f"  Signal: {f_tone/1e3:.0f} kHz tone")
    print(f"  Duration: {duration} seconds")
    print(f"  Samples: {num_samples}")
    
    # Setup streamer
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = channels
    tx_streamer = usrp.get_tx_stream(st_args)
    
    # Setup metadata
    metadata = uhd.types.TXMetadata()
    metadata.start_of_burst = True
    metadata.end_of_burst = True
    metadata.has_time_spec = False
    
    print("\nTransmitting signal...")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Send the signal 3 times
        for i in range(3):
            print(f"Transmission {i+1}/3...")
            
            # Reset metadata for each transmission
            metadata.start_of_burst = True
            
            # Send samples
            tx_streamer.send(signal, metadata)
            
            # Wait between transmissions
            time.sleep(1.0)
            
        print("\n✅ Transmission complete!")
        print("\nNotes:")
        print("- If testing with spectrum analyzer, look for peak at 2.45 GHz ± 100 kHz")
        print("- If testing with another USRP, it should receive this signal")
        print("- Try increasing gain if signal is too weak")
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    except Exception as e:
        print(f"\n❌ Error during transmission: {e}")
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test USRP transmission")
    parser.add_argument("--addr", default="192.168.10.3",
                        help="USRP IP address (TX)")
    args = parser.parse_args()
    
    test_simple_tx(args.addr)