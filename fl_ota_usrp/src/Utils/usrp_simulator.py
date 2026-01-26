"""
USRP Simulator for testing without hardware
Replaces real USRP operations with simulated ones
"""

import numpy as np
import time


class USRP_X310_Simulator:
    """Simulated USRP for testing without hardware"""
    
    def __init__(self, addr):
        self.addr = addr
        print(f"[SIMULATOR] USRP initialized at {addr}")
        self.tx_config = {}
        self.rx_config = {}
        
    def set_tx(self, freq, samprate, gain, channel, antenna):
        """Simulate TX configuration"""
        self.tx_config = {
            'freq': freq,
            'samprate': samprate, 
            'gain': gain,
            'channel': channel,
            'antenna': antenna
        }
        print(f"[SIMULATOR] TX configured: {freq/1e9:.2f} GHz, gain={gain} dB")
        
    def set_rx(self, freq, samprate, gain, channel, antenna):
        """Simulate RX configuration"""
        self.rx_config = {
            'freq': freq,
            'samprate': samprate,
            'gain': gain, 
            'channel': channel,
            'antenna': antenna
        }
        print(f"[SIMULATOR] RX configured: {freq/1e9:.2f} GHz, gain={gain} dB")
        
    def grad_to_wave(self, grads, amplitude, sps):
        """Simulate gradient to waveform conversion"""
        # Simple simulation: just scale and repeat
        waveform = np.repeat(grads * amplitude, sps).astype(np.complex64)
        scale = np.max(np.abs(grads))
        print(f"[SIMULATOR] Encoded {len(grads)} gradients to {len(waveform)} samples")
        return waveform, scale
        
    def tx_signal(self, waveform, repeat=False):
        """Simulate signal transmission"""
        print(f"[SIMULATOR] Transmitting {len(waveform)} samples...")
        # Simulate transmission delay
        time.sleep(0.1)
        # Store for potential loopback
        self.last_tx = waveform
        print(f"[SIMULATOR] Transmission complete")
        
    def rx_signal(self, num_samps):
        """Simulate signal reception"""
        print(f"[SIMULATOR] Receiving {num_samps} samples...")
        # Simulate with noise and multiple signals
        # In real OTA, signals would add naturally
        signal = np.random.randn(num_samps) + 1j * np.random.randn(num_samps)
        signal = signal.astype(np.complex64) * 0.1
        time.sleep(0.1)
        print(f"[SIMULATOR] Reception complete")
        return signal
        
    @staticmethod
    def wave_to_grad(wave, amplitude, sps, csi, scale):
        """Simulate waveform to gradient conversion"""
        # Simple simulation: downsample and scale back
        grads = wave[::sps].real / amplitude
        print(f"[SIMULATOR] Decoded {len(grads)} gradients from waveform")
        return grads


def csi_estimate_simulator(device_tx, device_rx):
    """Simulate CSI estimation"""
    # Return random complex channel coefficient
    csi = np.random.randn() + 1j * np.random.randn()
    csi = csi / np.abs(csi)  # Normalize
    print(f"[SIMULATOR] CSI estimated: {np.abs(csi):.3f} ∠ {np.angle(csi)*180/np.pi:.1f}°")
    return csi