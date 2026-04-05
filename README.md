# Over-the-Air Computation for Federated Learning

**ELEC498 — Queen’s University**

Ordinary federated learning ships each client’s update to a server, then averages on the server. **Over-the-air (OTA) aggregation** is a different idea: if several radios transmit on the same frequency at the same time, the channel *sums* their signals through superposition. In principle you can get a weighted sum in one shot instead of N separate uploads. This provides some interesting points for exploring latency and bandwith improvements.

Stack:

- **Simulation path (default):** [Flower](https://flower.ai/) coordinates clients; a custom strategy applies an OTA-style channel model when aggregating gradient deltas.
- **Hardware path (optional):** USRP X310 radios—one RX at the “server,” multiple TX at clients—timed so transmit and capture line up. Channel estimation runs first; then payload rounds use the same pilot framing so the receiver can find the burst.

**Dataset:** BloodMNIST (8-class medical imaging tiles). **Model in the current entrypoint:** a small CNN (see `fl_implementation/src/model.py`).

---

## What’s in the box

| Area | Where to look |
|------|----------------|
| FL orchestration | `fl_implementation/main.py` |
| Flower client / server wiring | `fl_implementation/src/client.py`, `server.py` |
| OTA aggregation (sim + USRP hooks) | `fl_implementation/src/ota_strategy.py` |
| USRP TX/RX helpers | `fl_implementation/src/usrp_x310.py`, `usrp_txrx.py` |
| Live metrics (ZMQ + Streamlit) | `fl_implementation/src/zmq_publisher.py`, `live_dashboard/` |

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
