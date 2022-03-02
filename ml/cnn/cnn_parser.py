import logging
import dpkt.ethernet
import numpy as np

from ml.common import int_list_from_bytes, display_resized_image
from ml.file_parser import Parser


class CNNParser(Parser):
    def __init__(self, filename, image_size=28, max_count=50000, shuffle=False, display=False):
        super().__init__(filename)
        self.image_size = image_size
        self.image_byte_count = self.image_size * self.image_size

        self.max_count = max_count
        self.shuffle = shuffle
        self.display = display

    def get_packet_count(self):
        return self.reader.snaplen

    def __generate_image(self, ethernet_data):
        ethernet_data = int_list_from_bytes(bytes(ethernet_data))
        data_length = len(ethernet_data)
        if data_length > self.image_byte_count:
            ethernet_data = np.array(ethernet_data[:self.image_byte_count], dtype=np.uint8)
        else:
            ethernet_data = np.array(ethernet_data, dtype=np.uint8)
            ethernet_data = np.pad(ethernet_data, (0, self.image_byte_count - data_length), 'constant',
                                   constant_values=0)

        image = ethernet_data.reshape((self.image_size, self.image_size))

        if self.display:
            display_resized_image(self.fname, image)

        return image

    def try_generate_image(self, idx, pkt):
        try:
            eth = dpkt.ethernet.Ethernet(pkt)
            ip = eth.data
            return self.__generate_image(ip)
        except dpkt.dpkt.Error as e:
            logging.warning(f"Could not unpack a packet number: {idx}. Error: {e}")

    def __get_random_packet_indexes(self, max_count=0):
        """
        Generates array of indexes from range <0,number of packets)
        Using rng.choice with replace=False to avoid repetitions

        :param max_count: max indexes to select
        :return: np array of indexes
        """
        rng = np.random.default_rng()
        count_to_read = min(max_count, self.get_packet_count())
        packet_indexes = rng.choice(self.get_packet_count(), count_to_read, replace=False)
        packet_indexes = np.sort(packet_indexes)

        return packet_indexes

    def parse_file_shuffle(self):
        packet_indexes = self.__get_random_packet_indexes(self.max_count)
        rand_idx = 0

        for idx, (ts, pkt) in enumerate(self.reader):
            if rand_idx < min(self.max_count, self.get_packet_count()):
                if idx == packet_indexes[rand_idx]:
                    yield self.try_generate_image(idx, pkt)
                    rand_idx += 1
            else:
                break

    def parse_file_normal(self, max_count=None):
        for idx, (ts, pkt) in enumerate(self.reader):
            if max_count is None or idx <= max_count:
                yield self.try_generate_image(idx, pkt)
            else:
                break

    def parse_file(self):
        if self.shuffle:
            yield from self.parse_file_shuffle()
        else:
            yield from self.parse_file_normal()
