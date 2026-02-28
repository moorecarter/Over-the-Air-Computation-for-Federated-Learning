import numpy as np
from gnuradio import gr

class channel_estimator(gr.sync_block):
    """
    Channel Estimator Block for Analog AirComp
    1. Detects the 'Known Sequence' via correlation.
    2. Estimates the Phase Offset at the peak.
    3. Prints the correction needed.
    """
    def __init__(self, known_sequence_coords=[1+1j, -1+1j]):
        gr.sync_block.__init__(self,
            name="Channel Estimator",
            in_sig=[np.complex64],
            out_sig=None) # Sink block (no output)

        # Convert input list to numpy array
        self.known_seq = np.array(known_sequence_coords, dtype=np.complex64)
        self.seq_len = len(self.known_seq)
        
        # State machine
        self.cooldown = 0
        self.threshold = 50.0 # Adjust based on your signal strength!

    def work(self, input_items, output_items):
        in0 = input_items[0]
        n_samples = len(in0)
        
        # If we just detected a packet, wait a bit before looking for another
        if self.cooldown > 0:
            self.cooldown -= n_samples
            return n_samples

        # 1. CROSS-CORRELATION
        # We look for the known sequence inside the incoming buffer
        # (This is computationally expensive, keep sample rate low or buffer small)
        if n_samples > self.seq_len:
            corr = np.correlate(in0, self.known_seq, mode='valid')
            peak_idx = np.argmax(np.abs(corr))
            peak_val = np.abs(corr[peak_idx])

            # 2. DETECTION TRIGGER
            if peak_val > self.threshold:
                # We found the packet!
                
                # Get the complex value at the peak
                # This complex value = (Channel H) * (Energy of Sequence)
                complex_peak = corr[peak_idx]
                
                # 3. CALCULATE PHASE
                # Angle of the peak tells us the rotation of the channel
                phase_rad = np.angle(complex_peak)
                phase_deg = np.degrees(phase_rad)
                
                print(f"Packet Detected! Peak: {peak_val:.1f}")
                print(f" -> Measured Channel Phase: {phase_deg:.2f} degrees")
                print(f" -> FEEDBACK TO CLIENT: Rotate by {-phase_deg:.2f} degrees")
                print("-" * 30)
                
                # Set cooldown to avoid double-triggering on the same packet
                self.cooldown = 2000 

        return n_samples
