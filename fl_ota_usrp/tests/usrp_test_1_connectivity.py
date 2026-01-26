#!/usr/bin/env python3
"""
USRP Test 1: Basic Connectivity
Test that we can connect to and configure USRPs
"""

import uhd
import numpy as np
import argparse
import sys

def test_usrp_connectivity(addr):
    """Test basic USRP connection and configuration"""
    print(f"=== USRP TEST 1: Connectivity ===")
    print(f"Testing USRP at {addr}\n")
    
    try:
        # Create USRP object
        print("Creating USRP object...")
        usrp = uhd.usrp.MultiUSRP(f"addr={addr}")
        print("✅ USRP object created")
        
        # Get device info
        print("\nDevice Information:")
        print(f"  - Device: {usrp.get_mboard_name(0)}")
        print(f"  - Serial: {usrp.get_mboard_eeprom(0).get('serial', 'N/A')}")
        
        # Check clock sources
        print("\nClock Sources:")
        clock_sources = usrp.get_clock_sources(0)
        print(f"  Available: {clock_sources}")
        current_clock = usrp.get_clock_source(0)
        print(f"  Current: {current_clock}")
        
        # Check time sources
        print("\nTime Sources:")
        time_sources = usrp.get_time_sources(0)
        print(f"  Available: {time_sources}")
        current_time = usrp.get_time_source(0)
        print(f"  Current: {current_time}")
        
        # Test RX configuration
        print("\nTesting RX configuration...")
        usrp.set_rx_rate(1e6)
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(2.45e9), 0)
        usrp.set_rx_gain(30, 0)
        usrp.set_rx_antenna("RX2", 0)
        
        actual_rate = usrp.get_rx_rate()
        actual_freq = usrp.get_rx_freq(0)
        actual_gain = usrp.get_rx_gain(0)
        actual_ant = usrp.get_rx_antenna(0)
        
        print(f"  Rate: {actual_rate/1e6:.1f} MHz")
        print(f"  Freq: {actual_freq/1e9:.3f} GHz")
        print(f"  Gain: {actual_gain} dB")
        print(f"  Antenna: {actual_ant}")
        print("✅ RX configuration successful")
        
        # Test TX configuration
        print("\nTesting TX configuration...")
        usrp.set_tx_rate(1e6)
        usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(2.45e9), 0)
        usrp.set_tx_gain(15, 0)
        usrp.set_tx_antenna("TX/RX", 0)
        
        actual_rate = usrp.get_tx_rate()
        actual_freq = usrp.get_tx_freq(0)
        actual_gain = usrp.get_tx_gain(0)
        actual_ant = usrp.get_tx_antenna(0)
        
        print(f"  Rate: {actual_rate/1e6:.1f} MHz")
        print(f"  Freq: {actual_freq/1e9:.3f} GHz")
        print(f"  Gain: {actual_gain} dB")
        print(f"  Antenna: {actual_ant}")
        print("✅ TX configuration successful")
        
        print("\n✅ ALL TESTS PASSED!")
        print(f"USRP at {addr} is ready for use")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check USRP is powered on")
        print("2. Check network cable is connected")
        print("3. Check IP address is correct")
        print("4. Try: uhd_find_devices")
        print("5. Try: ping", addr)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test USRP connectivity")
    parser.add_argument("--addr", default="192.168.10.2", 
                        help="USRP IP address")
    args = parser.parse_args()
    
    success = test_usrp_connectivity(args.addr)
    sys.exit(0 if success else 1)