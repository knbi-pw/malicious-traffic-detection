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
    count = 10000
    x, y = load_data(args.data, count=None, size_x=IMAGE_SIZE, size_y=LABEL_SIZE)
    x = x.reshape(x.shape[0], IMAGE_WIDTH, IMAGE_WIDTH, 1)

    model = keras.models.load_model(args.model)
    print("Model loaded!")

    print("Begin prediction...")
    start = time.time()
    # for i in range(count):
    #     predicted = model.predict(np.array([x[i]]))
    #     print(predicted)
    predicted = model.predict(x)
    prediction_argmax = predicted.argmax(axis=1)
    print(f"Num of equal predictions {sum(prediction_argmax==y.reshape(y.shape[0]))}/{y.shape[0]}")

    end = time.time() - start
    print(f"Prediction of {count} imgs took: {end:.2f}s")
    print(f"Prediction single took: {end / count:.2f}s = {1000*end/count}ms = {1000000*end/count}μs")


if __name__ == "__main__":
    main()
