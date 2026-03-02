import uhd 
import numpy as np 
import time


class USRP_X310():
    def __init__(
            self, 
            ip_addr: str = "192.168.40.2", 
        ):
        self.usrp = uhd.usrp.MultiUSRP(f"addr={ip_addr}")
        self.ip_addr = ip_addr
        self.tx_channels = {}
        self.rx_channels = {}
        self._rx_streamer = None
        self._tx_streamer = None
        print("Connected to:", self.usrp.get_mboard_name())

    def set_rx(
            self,
            freq: float,    # Center Freq (Hz)
            samprate: float = 1e6, # Sample rate( S/s)
            gain: int = 0, # RX Gain
            channel: int = 0, # Channel (keep 0)
            antenna="RX2",   # Change to where antenna is 
            lo_offset: float = 0.0
        ):
        self.usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("B:0"), channel)
        self.usrp.set_rx_rate(samprate, channel)
        self.usrp.set_rx_freq(uhd.types.TuneRequest(freq, lo_offset), channel)
        self.usrp.set_rx_gain(gain, channel)
        self.usrp.set_rx_antenna(antenna, channel)
        self.usrp.set_rx_bandwidth(samprate, channel)
        self.rx_channels[channel] = {"freq": freq, "rate": samprate, "gain": gain, "antenna": antenna}
        # Cache RX streamer (creating new ones each call causes segfaults in UHD 4.x)
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [channel]
        self._rx_streamer = self.usrp.get_rx_stream(st_args)
        print("\n--- RX DIAGNOSTICS ---")
        print("RX antennas available :", self.usrp.get_rx_antennas())
        print("RX antenna selected   :", self.usrp.get_rx_antenna())
        print("Requested RX freq     :", freq)
        print("Actual RX freq        :", self.usrp.get_rx_freq())
        print("Requested RX rate     :", samprate)
        print("Actual RX rate        :", self.usrp.get_rx_rate())
        print("Requested RX gain     :", gain)
        print("Actual RX gain        :", self.usrp.get_rx_gain())
        print("RX bandwidth          :", self.usrp.get_rx_bandwidth())
        print("----------------------\n")
        print(f"RX channel {channel}, antenna {antenna} configured. ")

    def set_tx(
            self,
            freq: float,    # Center Freq (Hz)
            samprate: float = 1e6,  # Sample rate( S/s)
            gain: int = 0, # TX gain
            channel: int = 0,   # Channel (keep 0)
            antenna="TX/RX",
            lo_offset: float = 0.0
        ):
        self.usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("B:0"), channel)
        self.usrp.set_tx_rate(samprate, channel)
        self.usrp.set_tx_freq(uhd.types.TuneRequest(freq, lo_offset), channel)
        self.usrp.set_tx_gain(gain, channel)
        self.usrp.set_tx_antenna(antenna, channel)
        self.tx_channels[channel] = {"freq": freq, "rate": samprate, "gain": gain, "antenna": antenna}
        # Cache TX streamer (creating new ones each call causes segfaults in UHD 4.x)
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [channel]
        self._tx_streamer = self.usrp.get_tx_stream(st_args)
        print("\n--- TX DIAGNOSTICS ---")
        print("TX antennas available :", self.usrp.get_tx_antennas())
        print("TX antenna selected   :", self.usrp.get_tx_antenna())
        print("Requested TX freq     :", freq)
        print("Actual TX freq        :", self.usrp.get_tx_freq())
        print("Requested TX rate     :", samprate)
        print("Actual TX rate        :", self.usrp.get_tx_rate())
        print("Requested TX gain     :", gain)
        print("Actual TX gain        :", self.usrp.get_tx_gain())
        print("TX gain range :", self.usrp.get_tx_gain_range())
        print("TX bandwidth          :", self.usrp.get_tx_bandwidth())
        print("----------------------\n")
        print(f"TX channel {channel}, antenna {antenna} configured.")

    def set_clk(self, clk_source: str = "external", time_source: str = "external"):
            self.usrp.set_clock_source(clk_source)
            self.usrp.set_time_source(time_source)
            print("External clock & PPS enabled, time reset to 0.")
    def rx_signal(
            self,
            num_samps: int = 10000,
            start_time: uhd.libpyuhd.types.time_spec = None
        ):
        try:
            if not self.rx_channels or self._rx_streamer is None:
                raise RuntimeError("No RX channels configured. Call set_rx() first.")
            streamer = self._rx_streamer
            metadata = uhd.types.RXMetadata()
            recv_buffer = np.zeros((1, streamer.get_max_num_samps()), dtype=np.complex64)
            
            # Schedule Rx of num_samps samples exactly 3 seconds from last PPS
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps
            stream_cmd.stream_now = True if start_time is None else False
            stream_cmd.time_spec = uhd.libpyuhd.types.time_spec(2.0) if start_time is None else start_time 
            streamer.issue_stream_cmd(stream_cmd)

            samples = np.zeros(num_samps, dtype=np.complex64)
            idx = 0
            timeout = 3.0 if start_time is None else 3.0 + start_time.get_real_secs()
            while idx < num_samps:
                n = streamer.recv(recv_buffer, metadata, timeout)
                if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                    print(f"UHD Error: {metadata.error_code}")
                    return None
                if n > 0:
                    end = min(idx + n, num_samps)
                    samples[idx:end] = recv_buffer[0][0:end - idx]
                    idx = end

            return samples
        except Exception as e:
            print("Error receiving signal:", e)
            return None

    def tx_signal(
            self, 
            waveform: np.ndarray, 
            repeat: bool = False,
            timeout: float = 1.0,
            start_time: uhd.libpyuhd.types.time_spec = None,
        ):
        try:
            if not self.tx_channels or self._tx_streamer is None:
                raise RuntimeError("No TX channels configured. Call set_tx() first.")
            streamer = self._tx_streamer
            metadata = uhd.types.TXMetadata()
            metadata.start_of_burst = True
            metadata.end_of_burst = False
            metadata.has_time_spec = start_time is not None
            if start_time is not None:
                metadata.time_spec = start_time

            if waveform.ndim == 1:
                waveform = waveform.reshape(1, -1)

            num_samps = waveform.shape[-1]
            total_sent = 0
            while total_sent < num_samps:
                chunk = waveform[:, total_sent:]
                sent = streamer.send(chunk, metadata, timeout)
                total_sent += sent
                metadata.start_of_burst = False
                metadata.has_time_spec = False

            # Signal end of burst
            metadata.end_of_burst = True
            streamer.send(np.zeros((1, 0), dtype=np.complex64), metadata, timeout)

            return num_samps                
            
        except Exception as e:
            print("Error transmitting signal:", e)
            return None
    @staticmethod
    def normalize_grads(grads: np.ndarray, eps = 1e-12):
        scale = np.max(np.abs(grads)) + eps
        return grads / scale, scale
    @staticmethod
    def _upsample(symbols, sps):
        # Pad in between with zeros
        upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
        upsampled[::sps] = symbols
        return upsampled
    
    @staticmethod
    def rrc_filter(sps, beta=0.35, num_taps=101):
        t = np.arange(num_taps) - (num_taps-1)//2
        t = t.astype(np.float32)

        # Avoid division by zero
        h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps)
        denom = 1 - (2*beta*t/sps)**2
        h[denom==0] = np.pi/4 * np.sinc(1/(2*beta))
        h /= denom

        # Normalize energy
        h /= np.sqrt(np.sum(h**2))
    
        return h.astype(np.float32)
    @staticmethod
    def grad_to_wave(grads: np.ndarray, amplitude: float, csi: complex = 1.0):
        # Normalize : Re->IQ : Upsample : RRC : waveform = RRC ⓧ Upsample

        # Normalize
        # normalized, scale = self.normalize_grads(grads=grads) 
        
        # Re-> IQ
        symbols = amplitude * grads.astype(np.complex64) / csi

        # Upsample for RRC
        # upsampled = self._upsample(symbols=symbols, sps=sps)

        # Perform RRC
        # rrc_filter = USRP_X310.rrc_filter(sps=sps)
        # rrc_waveform = np.convolve(upsampled, rrc_filter, mode='same')

        return symbols

    @staticmethod
    def wave_to_grad(waveform: np.ndarray, amplitude: float):
        # Reverse RRC
        # rrc_filter = USRP_X310.rrc_filter(sps=sps).astype(np.complex64)
        # rx_matched = np.convolve(wave, rrc_filter[::-1], mode='same')
                
        # Downsamp
        # rx_symbols = rx_matched[::sps]
        # grads = (rx_symbols / csi) / amplitude * scale
        grads = waveform.astype(np.complex64) / amplitude

        return np.real(grads)


    

        