import argparse
import csv
import logging

from ml.cnn.cnn_parser import CNNParser
from ml.pcap_processor import PcapProcessor


def csv_to_dict(csv_fname: str):
    dictionary = {}
    with open(csv_fname, "r") as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip the first row
        for row in reader:
            if row:
                if row[2]:
                    dictionary[f"{row[0]}.pcap"] = 1
                else:
                    dictionary[f"{row[0]}.pcap"] = 0

    return dictionary


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script creates binary files containing labels and images generated from the pcaps')
    parser.add_argument('-c', '--csv', type=str, help="Path to the CSV with hash names of the pcaps", required=True)
    parser.add_argument('-d', '--data', type=str, help="Directory with pcaps", required=True)
    parser.add_argument('-o', '--output', type=str, help="Directory to save the ubytes", required=True)
    parser.add_argument('-n', '--count', type=str, help="Pcap count to read", required=True)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()
    pcaps_data = csv_to_dict(args.csv)

    processor = PcapProcessor(parser=CNNParser,
                              result_dir=args.output,
                              pcaps_data=pcaps_data,
                              max_pcap_count=int(args.count),
                              input_dir=f"{args.data}\\")

    processor.process()


if __name__ == "__main__":
    main()
