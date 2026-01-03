import uhd # type: ignore
import numpy as np # type: ignore
import time


class USRP_X310():
    def __init__(
            self, 
            ip_addr: str = "192.168.40.2", 
        ):
        self.usrp = uhd.usrp.MultiUSRP(f"addr={ip_addr},type=x310")
        self.tx_channels = {}
        self.rx_channels = {}
        print("Connected to:", self.usrp.get_usrp_name())

    def set_rx(
            self,
            freq: float, 
            rate: float = 1e6, 
            gain: int = 10, 
            channel: int = 0, 
            antenna="RX2"
        ):
        self.usrp.set_rx_rate(rate, channel)
        self.usrp.set_rx_freq(uhd.types.TuneRequest(freq), channel)
        self.usrp.set_rx_gain(gain, channel)
        self.usrp.set_rx_antenna(antenna, channel)
        self.rx_channels[channel] = {"freq": freq, "rate": rate, "gain": gain, "antenna": antenna}
        print(f"RX channel {channel} configured. ")

    def set_tx(
            self,
            freq: float, 
            rate: float = 1e6, 
            gain: int = 10, 
            channel: int = 0,     
            antenna="TX/RX"       
        ):
        self.usrp.set_tx_rate(rate, channel)
        self.usrp.set_tx_freq(uhd.types.TuneRequest(freq), channel)
        self.usrp.set_tx_gain(gain, channel)
        self.usrp.set_tx_antenna(antenna, channel)
        self.tx_channels[channel] = {"freq": freq, "rate": rate, "gain": gain, "antenna": antenna}

        print(f"TX channel {channel} configured.")

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
            time_at_last_pps = self.usrp.get_time_last_pps().get_real_secs()
            while time_at_last_pps == self.usrp.get_time_last_pps().get_real_secs():
                time.sleep(0.1) # keep waiting till it happens- if this while loop never finishes then the PPS signal isn't there
            self.usrp.set_time_next_pps(uhd.libpyuhd.types.time_spec(0.0))

            # Schedule Rx of num_samps samples exactly 3 seconds from last PPS
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps
            stream_cmd.stream_now = True if start_time is None else True
            stream_cmd.time_spec = uhd.libpyuhd.types.time_spec(3.0) if start_time is None else start_time # set start time (try tweaking this)
            streamer.issue_stream_cmd(stream_cmd)

            # Receive Samples.  recv() will return zeros, then our samples, then more zeros, letting us know it's done
            waiting_to_start = start_time is not None # keep track of where we are in the cycle (see above comment)
            nsamps = 0
            i = 0
            samples = np.zeros(num_samps, dtype=np.complex64)
            while nsamps != 0 or waiting_to_start:
                nsamps = streamer.recv(recv_buffer, metadata)
                if nsamps and waiting_to_start:
                    waiting_to_start = False
                elif nsamps:
                    samples[i:i+nsamps] = recv_buffer[0][0:nsamps]
                i += nsamps
            print(len(samples))
            print(samples[0:10])
            return samples
        except Exception as e:
            print("Error receiving signal:", e)
            return None

    def tx_signal(
            self, 
            samples: np.ndarray, 
            repeat: bool = False,
            timeout: float = 1.0,
            start_time: uhd.libpyuhd.types.TimeSpec = None,
        ):
        try:
            if not hasattr(self, "rx_channels") or not self.rx_channels:
                raise RuntimeError("No RX channels configured. Call configure_rx() first.")
            chan = list(self.tx_channels.keys())[0]  # pick first configured channel
            st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            st_args.channels = [chan]
            metadata = uhd.types.RXMetadata()
            streamer = self.usrp.get_rx_stream(st_args)
            metadata = uhd.types.TXMetadata()
            metadata.start_of_burst = True
            metadata.end_of_burst = True

            if start_time is not None:
                metadata.has_time_spec = True
                metadata.time_spec = start_time

        # Make sure waveform is 2D: channels x samples
            if waveform.ndim == 1:
                waveform = waveform.reshape(1, -1)

            # Send waveform
            streamer.send_waveform(samples, channel=chan, repeat=repeat, timeout=timeout, tx_metadata=metadata)
            return waveform.shape[-1]                
            
        except Exception as e:
            print("Error transmitting signal:", e)
            return None

    

        