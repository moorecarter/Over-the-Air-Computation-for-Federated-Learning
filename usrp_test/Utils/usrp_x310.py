import uhd 
import numpy as np 
import time


class USRP_X310():
    def __init__(
            self, 
            ip_addr: str = "192.168.40.2", 
        ):
        self.usrp = uhd.usrp.MultiUSRP(f"addr={ip_addr}")
        self.tx_channels = {}
        self.rx_channels = {}
        print("Connected to:", self.usrp.get_usrp_name())

    def set_rx(
            self,
            freq: float,    # Center Freq (Hz)
            samprate: float = 1e6, # Sample rate( S/s)
            gain: int = 0, # RX Gain
            channel: int = 0, # Channel (keep 0)
            antenna="RX2"   # Change to where antenna is 
        ):
        self.usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("B:0"), channel)
        self.usrp.set_rx_rate(samprate, channel)
        self.usrp.set_rx_freq(uhd.types.TuneRequest(freq), channel)
        self.usrp.set_rx_gain(gain, channel)
        self.usrp.set_rx_antenna(antenna, channel)
        self.usrp.set_rx_bandwidth(samprate, channel)
        self.rx_channels[channel] = {"freq": freq, "rate": samprate, "gain": gain, "antenna": antenna}
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
            antenna="TX/RX"
        ):
        self.usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("B:0"), channel)
        self.usrp.set_tx_rate(samprate, channel)
        self.usrp.set_tx_freq(uhd.types.TuneRequest(freq), channel)
        self.usrp.set_tx_gain(gain, channel)
        self.usrp.set_tx_antenna(antenna, channel)
        self.tx_channels[channel] = {"freq": freq, "rate": samprate, "gain": gain, "antenna": antenna}
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

    def set_clk(self):
            self.usrp.set_clock_source("external")
            self.usrp.set_time_source("external")
            self.usrp.set_time_now(uhd.libpyuhd.types.TimeSpec(0.0))
            print("External clock & PPS enabled, time reset to 0.")

    def rx_signal(
            self,
            num_samps: int = 10000,
            start_time: uhd.libpyuhd.types.TimeSpec = None
        ):
        try:
            if not hasattr(self, "rx_channels") or not self.rx_channels:
                raise RuntimeError("No RX channels configured. Call configure_rx() first.")
            chan = list(self.rx_channels.keys())[0]  # pick first configured channel
            st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            st_args.channels = [chan]
            metadata = uhd.types.RXMetadata()
            streamer = self.usrp.get_rx_stream(st_args)
            recv_buffer = np.zeros((1, 1000), dtype=np.complex64)

            # Start Stream
            # copy the receive example above, everything up until # Start Stream

            # Wait for 1 PPS to happen, then set the time at next PPS to 0.0
            # time_at_last_pps = self.usrp.get_time_last_pps().get_real_secs()
            # while time_at_last_pps == self.usrp.get_time_last_pps().get_real_secs():
            #     time.sleep(0.1) # keep waiting till it happens- if this while loop never finishes then the PPS signal isn't there
            # self.usrp.set_time_next_pps(uhd.libpyuhd.types.time_spec(0.0))

            # Schedule Rx of num_samps samples exactly 3 seconds from last PPS
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps
            stream_cmd.stream_now = True if start_time is None else False
            # stream_cmd.time_spec = uhd.libpyuhd.types.time_spec(3.0) if start_time is None else start_time # set start time (try tweaking this)
            streamer.issue_stream_cmd(stream_cmd)

            # Receive Samples.  recv() will return zeros, then our samples, then more zeros, letting us know it's done
            # waiting_to_start = start_time is not None # keep track of where we are in the cycle (see above comment)
            # nsamps = 0
            # i = 0
            samples = np.zeros(num_samps, dtype=np.complex64)
            # while nsamps != 0 or waiting_to_start:
            #     nsamps = streamer.recv(recv_buffer, metadata)
            #     if nsamps and waiting_to_start:
            #         waiting_to_start = False
            #     elif nsamps:
            #         samples[i:i+nsamps] = recv_buffer[0][0:nsamps]
            #     i += nsamps
            idx = 0
            while idx < num_samps:
                n = streamer.recv(recv_buffer, metadata)
                if n > 0:
                    samples[idx:idx+n] = recv_buffer[:n]
                    idx += n

            return samples
            print(len(samples))
            print(samples[0:10])
            return samples
        except Exception as e:
            print("Error receiving signal:", e)
            return None

    def tx_signal(
            self, 
            waveform: np.ndarray, 
            repeat: bool = False,
            timeout: float = 1.0,
            start_time: uhd.libpyuhd.types.TimeSpec = None,
        ):
        try:
            if not hasattr(self, "tx_channels") or not self.tx_channels:
                raise RuntimeError("No TX channels configured. Call configure_tx() first.")
            chan = list(self.tx_channels.keys())[0]  # pick first configured channel
            st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            st_args.channels = [chan]
            streamer = self.usrp.get_tx_stream(st_args)
            metadata = uhd.types.TXMetadata()
            metadata.start_of_burst = True
            metadata.end_of_burst = True
            # metadata.has_time_spec = True
            # if start_time is not None:
            #     metadata.has_time_spec = True
            #     metadata.time_spec = start_time

        # Make sure waveform is 2D: channels x samples
            if waveform.ndim == 1:
                waveform = waveform.reshape(1, -1)

            # Send waveform
            streamer.send_waveform(waveform, channel=chan, repeat=repeat, timeout=timeout)
            return waveform.shape[-1]                
            
        except Exception as e:
            print("Error transmitting signal:", e)
            return None
    
    def _normalize_grads(self, grads: np.ndarray, eps = 1e-12):
        scale = np.max(np.abs(grads)) + eps
        return grads / scale, scale
    
    def _upsample(self, symbols, sps):
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

    def grad_to_wave(self, grads: np.ndarray, amplitude: float, sps: int):
        # Normalize : Re->IQ : Upsample : RRC : waveform = RRC ⓧ Upsample

        # Normalize
        normalized, scale = self._normalize_grads(grads=grads) 

        # Re-> IQ
        symbols = amplitude * normalized.astype(np.complex64)

        # Upsample for RRC
        upsampled = self._upsample(symbols=symbols, sps=sps)

        # Perform RRC
        rrc_filter = USRP_X310.rrc_filter(sps=sps)

        return np.convolve(upsampled, rrc_filter, mode='same'), scale


    def tx_pilot(self, amplitude: float, sps: int):
        pilot = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32)
        # Normalize, Re->IQ, Upsample, RRC
        waveform, _  = self.grad_to_wave(grads=pilot, amplitude=amplitude, sps=sps)
        self.tx_signal(waveform=waveform, repeat=False)
        return waveform

    def rx_pilot(self, num_samps: int, sps: float):
        rx_pilot = self.rx_signal(num_samps)
        # Reverse RRC Filter 
        rx_matched = np.convolve(rx_pilot, self._rrc_filter(sps=sps).astype(np.complex64)[::-1], mode='same')
        # Downsamp
        rx_symbols = rx_matched[::sps]
        return rx_symbols
    @staticmethod
    def wave_to_grad(wave: np.ndarray, amplitude: float, sps: int, csi, scale: float):
        # Reverse RRC
        rrc_filter = USRP_X310.rrc_filter(sps=sps).astype(np.complex64)
        rx_matched = np.convolve(wave, rrc_filter[::-1], mode='same')
                
        # Downsamp
        rx_symbols = rx_matched[::sps]
        grads = (rx_symbols / csi) / amplitude * scale

        return np.real(grads)






    

        