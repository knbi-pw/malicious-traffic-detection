import logging
from abc import ABC

import dpkt
import numpy as np
from ml.file_parser import Parser
from ml.common import int_list_from_bytes


class DeepParser(Parser, ABC):

    def __init__(self, filename, max_count=50000, shuffle=False):
        super().__init__(filename)
        self.n = 1024
        self.max_count = max_count
        self.shuffle = shuffle


    def trim_packets(self, tcp_payload):
        tcp_payload = int_list_from_bytes(bytes(tcp_payload))
        data_length = len(tcp_payload)
        if data_length > self.n:
            packet_bytes = np.array(tcp_payload[:self.n], dtype=np.uint8)
        else:
            packet_bytes = np.array(tcp_payload, dtype=np.uint8)
            packet_bytes = np.pad(packet_bytes, (0, self.n - data_length), 'constant',
                                  constant_values=0)
        return packet_bytes

    def parse_file(self):
        for idx, (ts, pkt) in enumerate(self.reader):
            if self.max_count is None or idx <= self.max_count:
                try:
                    eth = dpkt.ethernet.Ethernet(pkt)
                    ip = eth.data
                    tcp = ip.data
                    yield self.trim_packets(tcp)
                except dpkt.dpkt.Error as e:
                    logging.warning(f"Could not unpack a packet number: {idx}. Dpkt error: {e}")
                except Exception as e:
                    logging.warning(f"Could not unpack a packet number: {idx}. Error: {e}")

            else:
                break
