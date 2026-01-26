import numpy as np 
from Utils.usrp_x310 import USRP_X310
from Utils.csi_estimate import csi_estimate

import time
import uhd 
import threading

# Define params
RX_ADDR = "192.168.10.2"
TX1_ADDR = "192.168.10.3"
FC = 2.45e9
FS = 1e6
RX_GAIN = 30
RX_ANT = "RX2"
RX_CHAN = 0
TX1_GAIN = 15
TX1_ANT = "TX/RX"
TX1_CHAN = 0

def main():
    # Initialize usrps
    usrp_rx = USRP_X310(RX_ADDR)
    usrp_tx1 = USRP_X310(TX1_ADDR)

    #Setup usrps for transmit / receive
    usrp_rx.set_rx(freq=FC,samprate=FS, gain=RX_GAIN, channel=RX_CHAN, antenna=RX_ANT)
    usrp_tx1.set_tx(freq=FC,samprate=FS, gain=TX1_GAIN, channel=TX1_CHAN, antenna=TX1_ANT)

    # perform csi estimate
    print("Estimaing CSI...")
    h1 = csi_estimate(device_tx=usrp_tx1, device_rx=usrp_rx)
    # time for cooldown
    print(f"Done! CSI: {h1}")
    print("CSI Magnitude:", np.abs(h1), "CSI Phase (rad):", np.angle(h1))
    time.sleep(5)

    # params for message
    amplitude = 0.2
    sps = 100
    grads = np.array([-0.7, -0.4, -0.1, 0.1, 0.4, 0.7]) # random for now
    print("Original Gradient Values: ")
    print(grads)
    # prep wave for sending
    waveform, scale = usrp_tx1.grad_to_wave(grads=grads, amplitude=amplitude, sps=sps)
    num_samps = len(waveform)
    print("Sending Waveform: ")
    print(waveform)
    # Start RX in a thread BEFORE TX
    rx_symbols = None
    def rx_thread_fn():
        nonlocal rx_symbols
        rx_symbols = usrp_rx.rx_signal(num_samps)

    rx_thread = threading.Thread(target=rx_thread_fn)
    rx_thread.start()

    time.sleep(0.1)  

    # TX burst
    usrp_tx1.tx_signal(waveform=waveform, repeat=False)

    # Wait for RX to finish
    rx_thread.join()

    time.sleep(2)
    
    rx_grad_corrected = USRP_X310.wave_to_grad(wave=rx_symbols, amplitude=amplitude, sps=sps, csi=h1, scale=scale)

    tx_grads = grads 
    rx_grads = rx_grad_corrected

    error = rx_grads - tx_grads
    mse = np.mean(error**2)
    max_abs_error = np.max(np.abs(error))

    print("Recovered grads:", rx_grads)
    print("Error:", error)
    print("MSE:", mse)
    print("Max abs error:", max_abs_error)



if __name__ == "__main__":
    main()