# Over-the-Air Computation for Federated Learning

**ELEC498 — Queen’s University**

This project explores Over-The-Air (OTA) computation for Federated Learning (FL) and how it can boost training efficiency while maintaining data privacy. Traditional federated learning requires edge-devices to transmit their local model updates to a main server which are then aggregated to update the global model. OTA computation improves this by having edge devices transmit over the same frequency band, causing their waves to superimpose in the wireless channel. This allows the aggregated signal to directly update the model which limits communication latency, computation costs, and promotes data security by aggregating updates through a wireless channel. Implementation consists of multiple Universal Software Radio Peripheral X310 (USRP) devices that will synchronously send aggregated gradient updates to a central USRP acting as the main server. The central USRP will then decode these superimposed waveforms, update the global model, and distribute back to edge-devices. Successful completion of this project is broken down into five key milestones that all have defined objectives, timelines, and testing benchmarks. These milestones are outlined as single USRP literacy, expanding to multiple with OTA communication, FL with traditional networking, integrating FL with OTA communication, and performance comparison between the two approaches.

Stack:

- **Simulation path (default):** [Flower](https://flower.ai/) coordinates clients; a custom strategy applies an OTA-style channel model when aggregating gradient deltas.
- **Hardware path (optional):** USRP X310 radios—one RX at the “server,” multiple TX at clients—timed so transmit and capture line up. Channel estimation runs first; then payload rounds use the same pilot framing so the receiver can find the burst.

**Dataset:** BloodMNIST (8-class medical imaging tiles). **Model in the current entrypoint:** a small CNN (see `fl_implementation/src/model.py`).

---

## What’s in the Project

| Area | Where to look |
|------|----------------|
| FL orchestration | `fl_implementation/main.py` |
| Flower client / server wiring | `fl_implementation/src/client.py`, `server.py` |
| OTA aggregation (simulation & USRP hooks) | `fl_implementation/src/ota_strategy.py` |
| USRP TX/RX helpers | `fl_implementation/src/usrp_x310.py`, `usrp_txrx.py` |
| Live metrics (ZMQ + Streamlit) | `fl_implementation/src/zmq_publisher.py`, `live_dashboard/` |
| GNU Radio Test Flowgraphs | `gnr/` |

---

## Quick start (simulation)

From a shell:

```bash
cd fl_implementation
python -m venv .venv
```

Activate the venv (Windows PowerShell: `.\.venv\Scripts\Activate.ps1`), then:

```bash
pip install -r requirements.txt
python main.py --num-clients 3 --num-rounds 5 --partition iid --ota-channel moderate
```

Runs Flower with server + in-process clients on `localhost:8080`. Config and metrics land under `fl_implementation/outputs/<experiment_name>/` (`config.json`, `results.json`).

Use `python main.py --help` for every flag (image size, non-IID `alpha`, SNR override, USRP addresses, etc.).

---

## USRP run (short version)

Requires UHD / hardware drivers on the machine that talks to the radios. Then enable hardware in `main.py` with `--use-usrp` and pass `--server-usrp-addr` plus `--client-usrp-addrs` for each TX node. External clock/PPS is assumed in `init_all_usrps`—match that to how your bench is actually wired.

---

## Contributors

- Carter Moore  
- Trask Smith  
- Damon Clulow  
- Ethan Whitcher  

**Supervisor:** Dr. Ning Lu
