import numpy as np
from gnuradio import gr
import pmt

class blk(gr.sync_block):
    def __init__(self, sync_tag="ofdm_sync_found", len_tag="packet_len", packet_len=960):
        # packet_len = (FFT_Len + CP_Len) * Num_Symbols
        # e.g. (64 + 16) * 12 = 960
        gr.sync_block.__init__(self,
            name='Add Length Tag',
            in_sig=[np.complex64],
            out_sig=[np.complex64])
        self.sync_tag = sync_tag
        self.len_tag = len_tag
        self.packet_len = packet_len

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        # 1. Pass data through unchanged
        out[:] = in0

        # 2. Read existing tags
        tags = self.get_tags_in_window(0, 0, len(in0))
        
        # 3. Look for Sync tag and add Length tag
        for tag in tags:
            key = pmt.to_python(tag.key)
            if key == self.sync_tag:
                # We found the sync! Add the length tag at the same offset.
                self.add_item_tag(0, 
                                  tag.offset, 
                                  pmt.intern(self.len_tag), 
                                  pmt.from_long(self.packet_len))

        return len(out)
