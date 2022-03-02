import argparse
import logging

from common import load_json_config
from ml.dl.dl_parser import DeepParser
from ml.pcap_processor import PcapProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script creates binary files containing labels and images generated from the pcaps')
    parser.add_argument('-j', '--json', type=str,
                        help="Path to the JSON configuration file containing pcap paths with labels", required=True)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()
    config = load_json_config(args.json)
    json_data = load_json_config(config["json"])

    processor = PcapProcessor(parser=DeepParser,
                              result_dir=config["data_dir"],
                              pcaps_data=json_data,
                              max_pcap_count=config["pcap_count"])

    processor.process()


if __name__ == "__main__":
    main()
