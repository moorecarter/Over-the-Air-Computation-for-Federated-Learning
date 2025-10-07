# Over-the-Air-Computation-for-Federated-Learning

## Project Overview

This project explores Over-The-Air (OTA) computation for Federated Learning (FL) and how it can boost training efficiency while maintaining data privacy. Traditional federated learning requires edge-devices to transmit their local model updates to a main server which are then aggregated to update the global model. OTA computation improves this by having edge devices transmit over the same frequency band, causing their waves to superimpose in the wireless channel. This allows the aggregated signal to directly update the model which limits communication latency, computation costs, and promotes data security by aggregating updates through a wireless channel. Implementation consists of multiple Universal Software Radio Peripheral X310 (USRP) devices that will synchronously send aggregated gradient updates to a central USRP acting as the main server. The central USRP will then decode these superimposed waveforms, update the global model, and distribute back to edge-devices. Successful completion of this project is broken down into five key milestones that all have defined objectives, timelines, and testing benchmarks. These milestones are outlined as single USRP literacy, expanding to multiple with OTA communication, FL with traditional networking, integrating FL with OTA communication, and performance comparison between the two approaches.

## Contributors

- Carter Moore
- Trask Smith
- Damon Clulow
- Ethan Whitcher
- Supervised by Dr. Ning Lu

## Usage

```

```
