import uhd
import numpy as np
import time
import argparse
from usrp_txrx import *
from matplotlib import pyplot as plt
from simulate_txrx import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=1)
    parser.add_argument("--datafile", type=str, default="")
    parser.add_argument("--const-signal", action="store_true")
    parser.add_argument("--server-addr", type=str, default="192.168.10.3")
    parser.add_argument("--client-addrs", type=str, default=["192.168.10.2"], nargs="+")
    parser.add_argument("--use-usrps", action="store_true")
    parser.add_argument("--ota-sim-config", type=str, default="ideal", choices=["ideal", "good", "moderate", "challenging", "harsh"])
    return parser.parse_args()

# Helper functions to load and save data

def load_data(datafile):
    return np.load(datafile)

def save_data(data, datafile):
    np.save(datafile, data)

DURATION = 5.0     # 5 seconds
SAMPRATE = 1e6  # 1 MHz
# NUM_SAMPLES = int(SAMPRATE * DURATION)
NUM_SAMPLES = 1024
# Helper function to find mse between sent and received data, plot
# determine if rx is indeed the sum of the sent data
def calculate_mse(num_clients):
    received_data = load_data("rx_data.npy")
    sent_data = [load_data(f"client_signals_{i}.npy") for i in range(num_clients)]
    expected = np.mean(sent_data, axis=0)
    mse = np.mean(np.abs(expected - received_data)**2)
    return mse

def plot_data(sent_data, received_data):
    for client_idx, data in enumerate(sent_data):
        signal_avg = np.mean(np.real(data))
        plt.plot(np.real(data), label=f"Sent data client {client_idx}, mean: {signal_avg:.5f}")
    received_avg = np.mean(np.real(received_data))
    plt.plot(np.real(received_data), label=f"Received data, mean: {received_avg:.5f}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Sent and Received Data")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    args = parse_args()
    num_clients = args.num_clients
    server_addr = args.server_addr
    client_addrs = args.client_addrs
    datafile = args.datafile
    use_usrps = args.use_usrps
    # Init client signals
    if datafile:
        client_signals = load_data(datafile)
    elif args.const_signal:
        client_signals = [np.ones(NUM_SAMPLES) * np.random.choice([5, 2, 3, 1]) * 1e-3 for _ in range(num_clients)]
    else:
        client_signals = [np.random.uniform(low=-0.5, high=0.5, size=NUM_SAMPLES) for _ in range(num_clients)]
    for i in range(num_clients):
        save_data(client_signals[i], f"client_signals_{i}.npy")
    # Initialize all USRPs and set pps and time to 0
    if use_usrps:
        init_all_usrps(server_addr, client_addrs)
        print(f"Running test with server: {server_addr} clients: {client_addrs}")
        
        # Channel estimation
        csi = usrp_channel_estimation(num_clients, 0, server_addr, client_addrs)
        print(f"Channel state: {csi}")
        
        # Transmit and receive
        rx_data = usrp_transmit_and_receive(num_clients, 0, server_addr, client_addrs, client_signals, csi)
        print(f"RX data shape: {rx_data.shape}")
        # print(f"RX data: {rx_data}")

        # Save RX data
    else:
        simulated_result = test_ota_strategy(client_signals, args.ota_sim_config)
        rx_data = simulated_result

    save_data(rx_data, "rx_data.npy")
    # Calculate MSE
    mse = calculate_mse(num_clients)
    print(f"MSE: {mse}")
    # print(f"Received data shape: {rx_data.shape}")
    # print(client_signals[0].shape)
    # Plot data
    plot_data(client_signals, rx_data)


if __name__ == "__main__":
    main()