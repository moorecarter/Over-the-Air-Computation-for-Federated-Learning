#!/usr/bin/env python3
"""
USRP Test 3: Simple Reception
Test receiving signals on a USRP
"""

import uhd
import numpy as np
import argparse
import time

def test_simple_rx(addr):
    """Test basic signal reception"""
    print(f"=== USRP TEST 3: Simple Reception ===")
    print(f"Using USRP at {addr}\n")
    
    # Configuration
    freq = 2.45e9
    rate = 1e6
    gain = 30
    channels = [0]
    
    print("Configuration:")
    print(f"  Frequency: {freq/1e9:.3f} GHz")
    print(f"  Sample Rate: {rate/1e6:.1f} MHz")
    print(f"  RX Gain: {gain} dB")
    print(f"  Antenna: RX2\n")
    
    # Create USRP
    print("Initializing USRP...")
    usrp = uhd.usrp.MultiUSRP(f"addr={addr}")
    
    # Configure RX
    usrp.set_rx_rate(rate)
    for chan in channels:
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq), chan)
        usrp.set_rx_gain(gain, chan)
        usrp.set_rx_antenna("RX2", chan)
    
    # Setup streamer
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = channels
    rx_streamer = usrp.get_rx_stream(st_args)
    
    # Setup receive buffer
    max_samps_per_packet = rx_streamer.get_max_num_samps()
    recv_buffer = np.zeros(max_samps_per_packet, dtype=np.complex64)
    
    # Setup metadata
    metadata = uhd.types.RXMetadata()
    
    # Start streaming
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)
    
    print("Receiving signal...")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Receive for 5 seconds
        timeout = 5.0
        start_time = time.time()
        total_samples = 0
        max_magnitude = 0
        
        while (time.time() - start_time) < timeout:
            # Receive samples
            num_rx = rx_streamer.recv(recv_buffer, metadata)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                print(f"Error: {metadata.error_code}")
                
            if num_rx > 0:
                total_samples += num_rx
                
                # Calculate signal power
                magnitude = np.abs(recv_buffer[:num_rx])
                power_dbfs = 20 * np.log10(np.mean(magnitude) + 1e-12)
                max_mag = np.max(magnitude)
                
                if max_mag > max_magnitude:
                    max_magnitude = max_mag
                
                # Print status every second
                if int(time.time() - start_time) != int(time.time() - start_time - 0.1):
                    print(f"  Time: {time.time()-start_time:.1f}s, "
                          f"Samples: {total_samples}, "
                          f"Power: {power_dbfs:.1f} dBFS, "
                          f"Peak: {20*np.log10(max_magnitude+1e-12):.1f} dBFS")
        
        # Stop streaming
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx_streamer.issue_stream_cmd(stream_cmd)
        
        print(f"\n✅ Reception complete!")
        print(f"Total samples received: {total_samples}")
        print(f"Peak magnitude: {max_magnitude:.6f}")
        
        if max_magnitude > 0.01:
            print("\n✅ Strong signal detected!")
        elif max_magnitude > 0.001:
            print("\n⚠️  Weak signal detected - try increasing TX gain")
        else:
            print("\n⚠️  No significant signal detected")
            print("Check:")
            print("- TX is running on other USRP")
            print("- Cables are connected (if wired)")
            print("- Antennas are attached (if wireless)")
            print("- Frequency matches TX frequency")
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        
    except Exception as e:
        print(f"\n❌ Error during reception: {e}")
        return False
    
    finally:
        # Always stop streaming
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx_streamer.issue_stream_cmd(stream_cmd)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test USRP reception")
    parser.add_argument("--addr", default="192.168.10.2",
                        help="USRP IP address (RX)")
    args = parser.parse_args()
    
    test_simple_rx(args.addr)