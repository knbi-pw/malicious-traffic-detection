import dpkt
from abc import ABC, abstractmethod


class Parser(ABC):

    def __init__(self, filename):
        self.reader = dpkt.pcap.Reader(open(filename, 'rb'))
        self.fname = filename

    @abstractmethod
    def parse_file(self):
        pass


