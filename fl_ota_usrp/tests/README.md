# USRP Integration Testing Guide

Incremental tests to integrate USRP functionality step by step.

## Prerequisites

- USRP X310 devices connected to network
- UHD drivers installed (`uhd_find_devices` works)
- Python UHD bindings (`pip install uhd`)
- For wired tests: SMA cables and 30dB attenuators

## Test Sequence

### 1. Connectivity Test
**Purpose**: Verify basic USRP connection and configuration

```bash
python usrp_test_1_connectivity.py --addr 192.168.10.2
```

**Expected**: USRP info displayed, TX/RX configuration successful

### 2. Simple TX Test  
**Purpose**: Verify USRP can transmit signals

```bash
python usrp_test_2_simple_tx.py --addr 192.168.10.3
```

**Expected**: Transmits 100 kHz tone at 2.45 GHz

### 3. Simple RX Test
**Purpose**: Verify USRP can receive signals

```bash
# Run in separate terminal while TX is running
python usrp_test_3_simple_rx.py --addr 192.168.10.2
```

**Expected**: Shows received signal power

### 4. TX-RX Pair Test (Wired)
**Purpose**: Test communication between two USRPs

**Setup**: Connect TX to RX with SMA cable + 30dB attenuator

```bash
python usrp_test_4_tx_rx_pair.py --rx-addr 192.168.10.2 --tx-addr 192.168.10.3
```

**Expected**: Signal transmitted and received, SNR > 20 dB

### 5. CSI Estimation Test
**Purpose**: Test channel estimation (critical for OTA)

```bash
python usrp_test_5_csi_estimation.py --rx-addr 192.168.10.2 --tx-addr 192.168.10.3
```

**Expected**: Stable CSI measurements, improved gradient recovery with CSI

### 6. Gradient Transmission Test
**Purpose**: Test actual gradient encoding/decoding

```bash
python usrp_test_6_gradient_transmission.py --rx-addr 192.168.10.2 --tx-addr 192.168.10.3
```

**Expected**: Gradients successfully encoded, transmitted, and recovered

### 7. OTA Test (2 Clients)
**Purpose**: Test simultaneous transmission from multiple USRPs

**Requires**: 3 USRPs (1 RX, 2 TX)

```bash
python usrp_test_7_ota_simulation.py \
    --rx-addr 192.168.10.2 \
    --tx1-addr 192.168.10.3 \
    --tx2-addr 192.168.10.4
```

**Expected**: Both signals combine in air, aggregated gradient recovered

### 8. Full FL Test
**Purpose**: Complete federated learning round with USRPs

```bash
# Terminal 1 - Server
cd ..
python main.py --mode server --usrp-addr 192.168.10.2

# Terminal 2 - Client 1
python main.py --mode client --client-id 0 --usrp-addr 192.168.10.3

# Terminal 3 - Client 2  
python main.py --mode client --client-id 1 --usrp-addr 192.168.10.4
```

## Troubleshooting

### No USRP Found
```bash
uhd_find_devices
ping 192.168.10.2
```

### Low Signal / No Signal
- Check cable connections
- Verify attenuators (30dB for wired)
- Increase TX gain gradually
- Check frequency alignment

### High Error Rate
- Re-run CSI estimation
- Check for interference (2.45 GHz)
- Reduce distance (if wireless)
- Check synchronization timing

## Safety Notes

⚠️ **For Wired Tests**: Always use attenuators (20-30 dB) to protect RX
⚠️ **For Wireless**: Start with low TX gain (5-10 dB), increase gradually
⚠️ **Frequency**: 2.45 GHz is ISM band but check local regulations

## Next Steps

After all tests pass:
1. Replace synthetic data with real datasets
2. Implement power control
3. Add more clients (3+)
4. Test in different channel conditions
5. Optimize gradient encoding