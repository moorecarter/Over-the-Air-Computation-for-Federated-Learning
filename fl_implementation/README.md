# Federated Learning with OTA Aggregation Demo

Demonstrates Flower's utility as an orchestrator for federated learning with Over-The-Air (OTA) computation, using MedViTV2 on BloodMNIST.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Flower Server                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           FedAvgOTA Strategy                         │   │
│  │  • Coordinates training rounds                       │   │
│  │  • Simulates OTA channel effects (noise, fading)     │   │
│  │  • Aggregates model updates                          │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Client 0    │  │   Client 1    │  │   Client 2    │
│  (Hospital A) │  │  (Hospital B) │  │  (Hospital C) │
│               │  │               │  │               │
│  BloodMNIST   │  │  BloodMNIST   │  │  BloodMNIST   │
│  (partition)  │  │  (partition)  │  │  (partition)  │
│               │  │               │  │               │
│  MedViTV2     │  │  MedViTV2     │  │  MedViTV2     │
│  (local)      │  │  (local)      │  │  (local)      │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo (fastest settings)
python main.py --model medvit_tiny --img-size 28 --num-rounds 5

# Run with better model (slower)
python main.py --model medvit_small --img-size 224 --num-rounds 10
```

## Key Components

| File | Description |
|------|-------------|
| `main.py` | Entry point - run experiments |
| `src/model.py` | MedViTV2 architecture with KAN layers |
| `src/data.py` | BloodMNIST loading + federated partitioning |
| `src/ota_strategy.py` | OTA channel simulation + FedAvgOTA strategy |
| `src/client.py` | Flower client implementation |
| `src/server.py` | Server-side evaluation + strategy creation |

## Experiments

### 1. Compare Data Distributions

```bash
# IID (independent, identically distributed)
python main.py --partition iid --experiment-name exp_iid

# Non-IID (heterogeneous across clients)
python main.py --partition non_iid --alpha 0.5 --experiment-name exp_non_iid

# Pathological (each client sees only 2 classes)
python main.py --partition pathological --experiment-name exp_pathological
```

### 2. Compare OTA Channel Conditions

```bash
# Ideal channel (no noise, no fading)
python main.py --ota-channel ideal --experiment-name exp_ideal

# Good channel (high SNR, no fading)
python main.py --ota-channel good --experiment-name exp_good

# Moderate channel (medium SNR, Rician fading)
python main.py --ota-channel moderate --experiment-name exp_moderate

# Challenging channel (low SNR, Rayleigh fading)
python main.py --ota-channel challenging --experiment-name exp_challenging

# Harsh channel (worst case)
python main.py --ota-channel harsh --experiment-name exp_harsh
```

### 3. Scalability

```bash
# Few clients
python main.py --num-clients 3 --experiment-name exp_3clients

# Many clients
python main.py --num-clients 10 --experiment-name exp_10clients
```

## OTA Channel Presets

| Preset | SNR (dB) | Fading | Use Case |
|--------|----------|--------|----------|
| `ideal` | ∞ | None | Baseline comparison |
| `good` | 30 | None | Indoor, close range |
| `moderate` | 20 | Rician | Indoor, some obstacles |
| `challenging` | 15 | Rayleigh | Outdoor, mobile |
| `harsh` | 10 | Rayleigh | Worst case testing |

## Model Variants

| Model | Parameters | Notes |
|-------|------------|-------|
| `medvit_tiny` | ~1M | Fast iteration, demos |
| `medvit_small` | ~6M | Balanced |
| `medvit_base` | ~22M | Best accuracy |
| `medvit_small_no_kan` | ~6M | Ablation (no KAN layers) |

## BloodMNIST Dataset

- **Task**: Blood cell type classification
- **Classes**: 8 (basophil, eosinophil, erythroblast, etc.)
- **Train/Val/Test**: 11,959 / 1,712 / 3,421 images
- **Source**: [MedMNIST](https://medmnist.com/)

## Next Steps: Real OTA with USRP X310

This demo validates Flower's orchestration capabilities. For actual OTA:

1. **Phase 2**: Replace simulated channel with USRP loopback (wired)
2. **Phase 3**: Full wireless OTA aggregation

The `OTAChannelModel` class in `src/ota_strategy.py` can be swapped for real SDR code while keeping Flower's orchestration layer.

```python
# Future: Real OTA integration point
class RealOTAChannel:
    def __init__(self, usrp_config):
        self.usrp = setup_usrp(usrp_config)

    def aggregate_with_channel(self, client_updates, weights):
        # Transmit via USRP, receive aggregated signal
        return self.usrp.transmit_and_receive(client_updates)
```

## Project Structure

```
flwr/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── pyproject.toml          # Project config
├── README.md               # This file
├── src/
│   ├── __init__.py
│   ├── model.py            # MedViTV2 architecture
│   ├── data.py             # BloodMNIST + partitioning
│   ├── ota_strategy.py     # OTA simulation
│   ├── client.py           # Flower client
│   └── server.py           # Server strategy
├── configs/                # (optional) experiment configs
├── scripts/                # (optional) utility scripts
└── outputs/                # Experiment results
    └── <experiment_name>/
        ├── config.json
        └── results.json
```

## References

- [Flower](https://flower.ai/) - Federated learning framework
- [MedViTV2](https://github.com/Omid-Nejati/MedViTV2) - Vision transformer for medical imaging
- [MedMNIST](https://medmnist.com/) - Medical imaging benchmark
- [Over-the-Air Computation](https://arxiv.org/abs/2006.16703) - OTA aggregation concepts
