import argparse
import time

import numpy as np
from tensorflow import keras

from common import load_data

LABEL_SIZE = 1

IMAGE_WIDTH = 28
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH  # e.g. 28x28 for image and 1 byte for label


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="Path to the test data")
    ap.add_argument("-m", "--model", required=True, help="Path to the trained model")
    return ap.parse_args()


def main():
    args = parse_args()
    x, y = load_data(args.data, count=10000, size_x=IMAGE_SIZE, size_y=LABEL_SIZE)
    x = x.reshape(x.shape[0], IMAGE_WIDTH, IMAGE_WIDTH, 1)

    model = keras.models.load_model(args.model)
    print("Model loaded!")

    print("Begin prediction...")
    start = time.time()
    results = model.evaluate(x, y, batch_size=500)

    elapsed = time.time() - start
    print(f"Evaluation using {y.shape[0]} images took: {elapsed}")
    # print(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()
