import logging
import os
import time
from typing import Type, Dict

import numpy as np

from common import try_create_directory
from ml.file_parser import Parser


def handle_incorrect_write(b, train_f):
    if b != len(train_f):
        logging.error(f"written {b}/{len(train_f)} bytes")


def process_pcap(parser: Parser, result_fname_train_x, result_fname_train_y, result_fname_test_x, result_fname_test_y,
                 test_percentage=0.1, label=None):
    imgs = [img for img in parser.parse_file() if not isinstance(img, type(None))]
    imgs = np.array(imgs, dtype=object)
    np.random.shuffle(imgs)
    cut_index = int(test_percentage * len(imgs))
    train = imgs[cut_index:]
    test = imgs[:cut_index]

    with open(result_fname_train_x, "ab") as f:
        train_f = bytearray((train.flatten()).tolist())
        b = f.write(train_f)
        handle_incorrect_write(b, train_f)

    with open(result_fname_train_y, "ab") as f:
        b = f.write(bytearray([label] * len(train)))
        handle_incorrect_write(b, train)

    with open(result_fname_test_x, "ab") as f:
        test_f = bytearray((test.flatten()).tolist())
        b = f.write(test_f)
        handle_incorrect_write(b, test_f)

    with open(result_fname_test_y, "ab") as f:
        b = f.write(bytearray([label] * len(test)))
        handle_incorrect_write(b, test)


def create_output_path(directory, pcap_path, extension='ubyte'):
    pcap_path = os.path.basename(pcap_path)
    fname, _ = os.path.splitext(pcap_path)
    new_fname = f"{fname}.{extension}"
    return os.path.join(directory, new_fname)


class PcapProcessor:
    def __init__(self, parser: Type[Parser], result_dir: str, pcaps_data: Dict[str, int], max_pcap_count: int = None):
        try_create_directory(result_dir)

        self.parser = parser

        self.fname_train_images = create_output_path(result_dir, "train_images")
        self.fname_train_labels = create_output_path(result_dir, "train_labels")
        self.fname_test_images = create_output_path(result_dir, "test_images")
        self.fname_test_labels = create_output_path(result_dir, "test_labels")
        self.max_pcap_count = max_pcap_count
        self.pcaps_data = pcaps_data

    def process(self):
        for pcap_path in self.pcaps_data:
            logging.info(f"Processing {pcap_path}")

            start = time.time()

            parser = self.parser(filename=pcap_path, max_count=self.max_pcap_count, shuffle=True)
            process_pcap(parser, self.fname_train_images, self.fname_train_labels, self.fname_test_images,
                         self.fname_test_labels,
                         test_percentage=0.1, label=self.pcaps_data[pcap_path])
            logging.info(f"Processing {pcap_path} took: {time.time() - start}s")