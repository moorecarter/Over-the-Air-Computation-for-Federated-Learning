# Federated Learning with OTA Computation using USRP X310

This implementation demonstrates federated learning with over-the-air (OTA) gradient aggregation using USRP X310 software-defined radios.

## Architecture

```
┌──────────────────────────────────────────────────┐
│           Central Server (USRP RX)               │
│  • Coordinates FL rounds                         │
│  • Receives OTA-aggregated gradients             │
│  • Updates global model                          │
└─────────────────┬────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬─────────────┐
    │             │             │             │
    ▼             ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│Client 0 │  │Client 1 │  │Client 2 │  │Client N │
│USRP TX  │  │USRP TX  │  │USRP TX  │  │USRP TX  │
│Local    │  │Local    │  │Local    │  │Local    │
│Training │  │Training │  │Training │  │Training │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
```

## Key Features

- **OTA Aggregation**: All clients transmit gradients simultaneously; signals naturally sum in the air
- **Channel Estimation**: Periodic CSI estimation for channel compensation
- **Synchronized Transmission**: Precise timing coordination for simultaneous transmission
- **Flexible Architecture**: Single codebase with server/client modes

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify USRP connectivity
uhd_find_devices
```

## Usage

### Start Server (on machine with RX USRP)

```bash
python main.py --mode server \
    --usrp-addr 192.168.10.2 \
    --num-clients 3 \
    --num-rounds 10
```

### Start Clients (on machines with TX USRPs)

```bash
# Client 0
python main.py --mode client \
    --client-id 0 \
    --usrp-addr 192.168.10.3 \
    --server-addr <server_ip>

# Client 1  
python main.py --mode client \
    --client-id 1 \
    --usrp-addr 192.168.10.4 \
    --server-addr <server_ip>

# Client 2
python main.py --mode client \
    --client-id 2 \
    --usrp-addr 192.168.10.5 \
    --server-addr <server_ip>
```

## Configuration

Use a JSON config file for advanced settings:

```bash
python main.py --mode server --usrp-addr 192.168.10.2 --config configs/default_config.json
```

See `configs/default_config.json` for available parameters.

## Workflow

1. **Initialization**
   - Server starts and waits for client connections
   - Clients connect via control channel (TCP)
   - Channel state information (CSI) is estimated

2. **Training Rounds**
   - Server broadcasts global model to all clients
   - Clients perform local training on their data
   - Clients compute gradient updates
   - All clients transmit gradients simultaneously via USRP
   - Server receives OTA-aggregated signal
   - Server decodes and applies aggregated gradients
   - Process repeats for specified rounds

3. **Synchronization**
   - Precise timing ensures simultaneous transmission
   - CSI re-estimated periodically for channel changes

## Files

- `main.py` - Entry point with server/client modes
- `src/fl_usrp_server.py` - FL server with OTA reception
- `src/fl_usrp_client.py` - FL client with OTA transmission  
- `src/Utils/` - USRP utilities (from usrp_test)
- `configs/` - Configuration files

## Next Steps

1. **Replace Synthetic Data**: Integrate real datasets (e.g., BloodMNIST from fl_implementation)
2. **Advanced Channel Compensation**: Implement better CSI-based equalization
3. **Power Control**: Adjust transmission power based on channel conditions
4. **Fault Tolerance**: Handle client dropouts gracefully
5. **Performance Metrics**: Add accuracy evaluation and convergence tracking

## Notes

- Ensure all USRPs are time-synchronized (GPS or shared reference clock recommended)
- Test with wired connections first (using attenuators) before going wireless
- Monitor for interference in the 2.45 GHz ISM band