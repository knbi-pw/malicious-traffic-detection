import argparse
import logging
import os

import numpy as np
import pandas as pd
from tensorflow import keras

from ml.common import byte_arr_to_int_arr
from ml.cnn.lenet import LeNetModel
from train_cnn import read_json

IMAGE_WIDTH = 28
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH  # e.g. 28x28 for image and 1 byte for label


def read_data_from_single_file(fname, count=None):
    data = []
    stop_flag = False
    idx = 0

    with open(fname, 'rb') as f:
        while not stop_flag and (count is None or idx < count):
            img_data = f.read(IMAGE_SIZE)
            if len(img_data) == IMAGE_SIZE:
                data.append(byte_arr_to_int_arr(img_data))
                idx += 1
            else:
                stop_flag = True
    logging.info(f"{fname} loaded. count: {idx} imgs")
    return data


def extract_images_labels(data):
    img_data = [np.reshape(f_data, (IMAGE_WIDTH, IMAGE_WIDTH, 1)) for f_data in data]
    return img_data


def load_data(data_dir, count_per_file=None):
    data = []

    for filename in os.listdir(data_dir):
        logging.info(f"{filename} loaded.")
        imagePath = data_dir + "/" + filename
        data += read_data_from_single_file(imagePath, count_per_file)

    if data:
        return extract_images_labels(data)
    else:
        logging.warning("No data loaded")
        return None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="Path to the test data")
    ap.add_argument("-m", "--model", required=True, help="Path to the trained model")
    return ap.parse_args()


def main():
    args = parse_args()
    x = load_data(args.data, count_per_file=15)
    model = keras.models.load_model(args.model)
    predicted = model.predict(np.array([x[0]]))


if __name__ == "__main__":
    main()
