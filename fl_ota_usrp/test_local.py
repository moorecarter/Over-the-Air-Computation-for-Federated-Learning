#!/usr/bin/env python3
"""
Test FL+OTA locally without USRPs
Simulates the workflow with dummy USRP operations
"""

import subprocess
import time
import sys

def test_local_simulation():
    """Test with 1 server and 2 clients on same machine"""
    
    print("Starting FL+OTA Local Test (no USRPs)")
    print("=" * 50)
    
    # Start server in background
    print("Starting server...")
    server_cmd = [
        sys.executable, "main.py",
        "--mode", "server",
        "--usrp-addr", "dummy_server",  # Will need to handle in code
        "--num-clients", "2",
        "--num-rounds", "3"
    ]
    server_proc = subprocess.Popen(server_cmd)
    time.sleep(2)
    
    # Start client 0
    print("Starting client 0...")
    client0_cmd = [
        sys.executable, "main.py", 
        "--mode", "client",
        "--client-id", "0",
        "--usrp-addr", "dummy_client0",
        "--server-addr", "localhost"
    ]
    client0_proc = subprocess.Popen(client0_cmd)
    time.sleep(1)
    
    # Start client 1
    print("Starting client 1...")
    client1_cmd = [
        sys.executable, "main.py",
        "--mode", "client", 
        "--client-id", "1",
        "--usrp-addr", "dummy_client1",
        "--server-addr", "localhost"
    ]
    client1_proc = subprocess.Popen(client1_cmd)
    
    # Wait for completion
    print("\nRunning FL training...")
    print("Press Ctrl+C to stop\n")
    
    try:
        server_proc.wait()
        client0_proc.wait()
        client1_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping processes...")
        server_proc.terminate()
        client0_proc.terminate()
        client1_proc.terminate()

if __name__ == "__main__":
    test_local_simulation()