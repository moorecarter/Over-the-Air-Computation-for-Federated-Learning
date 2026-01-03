from Utils.usrp_x310 import USRP_X310
import numpy as np # type: ignore
import uhd # type: ignore
if __name__ == "__main__":
    usrp = USRP_X310()
    usrp.set_rx(freq=100e6) # 100 MHz
    usrp.set_tx(freq=100e6) # 100 MHz
    usrp.set_clk()

    # Sending arbirtary bit stream
    bits = np.array([0, 1, 1, 0, 1, 0])
    symbols = 2*bits - 1 # map to {-1, +1}
    waveform = symbols.astype(np.complex64)
    start = uhd.libpyuhd.types.TimeSpec(3.0)
    usrp.tx_signal(samples=waveform, start_time=start)

    # Recieve bit stream
    data = usrp.rx_signal()
