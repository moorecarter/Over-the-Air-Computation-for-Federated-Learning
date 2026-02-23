# Testing Guide for FL+OTA+USRP

## Phase 1: Local Simulation (No Hardware)

Test the entire workflow without USRPs:

```bash
cd fl_ota_usrp

# Set simulation mode
export FL_OTA_SIMULATION=true

# Terminal 1 - Start server
python main.py --mode server --usrp-addr sim_server --num-clients 2 --num-rounds 3

# Terminal 2 - Start client 0
python main.py --mode client --client-id 0 --usrp-addr sim_client0 --server-addr localhost

# Terminal 3 - Start client 1  
python main.py --mode client --client-id 1 --usrp-addr sim_client1 --server-addr localhost
```

Or use the automated test script:
```bash
export FL_OTA_SIMULATION=true
python test_local.py
```

## Phase 2: Wired USRP Test (With Hardware)

Test with real USRPs using cables and attenuators:

### Hardware Setup
1. Connect USRPs with SMA cables
2. Add 30dB attenuators to prevent overload
3. Server USRP: RX port connected
4. Client USRPs: TX ports connected via combiner

### Run Test
```bash
# Terminal 1 - Server (RX USRP)
python main.py --mode server --usrp-addr 192.168.10.2 --num-clients 2 --num-rounds 3

# Terminal 2 - Client 0 (TX USRP)
python main.py --mode client --client-id 0 --usrp-addr 192.168.10.3 --server-addr <server_ip>

# Terminal 3 - Client 1 (TX USRP)
python main.py --mode client --client-id 1 --usrp-addr 192.168.10.4 --server-addr <server_ip>
```

## Phase 3: Wireless Test

### Setup
1. Attach antennas to USRPs
2. Start with USRPs close together (1-2 meters)
3. Use lower TX gain initially (5-10 dB)

### Monitoring
Watch for:
- CSI estimation values
- Gradient transmission timing
- Aggregation accuracy

## Debugging Tips

### 1. Test USRP Connectivity First
```bash
# Find USRPs on network
uhd_find_devices

# Test individual USRP
uhd_usrp_probe --args="addr=192.168.10.2"
```

### 2. Test Gradient Encoding/Decoding
```python
# Test the core OTA mechanism from usrp_test
cd ../usrp_test
python src/main.py
```

### 3. Common Issues

**Connection refused**: 
- Check firewall settings
- Verify server is running
- Check IP addresses

**USRP not found**:
- Verify USRP IP configuration
- Check network connectivity
- Ensure UHD drivers installed

**Poor aggregation accuracy**:
- Increase TX gain carefully
- Check CSI estimation
- Verify synchronization timing

### 4. Incremental Testing
1. First test with 1 client only
2. Then test with 2 clients
3. Gradually increase to more clients

## Performance Metrics to Track

- **Convergence**: Model loss over rounds
- **Timing**: OTA transmission duration
- **Accuracy**: Gradient reconstruction error
- **Channel**: CSI stability over time

## Next Steps After Basic Testing

1. **Add Real Data**: Replace synthetic data with actual datasets
2. **Implement Metrics**: Add accuracy tracking and visualization
3. **Optimize Encoding**: Improve gradient-to-waveform mapping
4. **Channel Adaptation**: Adjust for changing channel conditions
5. **Scale Testing**: Test with more clients (3+)