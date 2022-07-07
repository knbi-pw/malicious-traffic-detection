from argparse import ArgumentParser

import numpy as np

from ml.cnn.lenet_pytorch import prepare_data_for_torch, LeNetTorchWrapper
from ml.image_batch_generator import ImageBatchGenerator


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_directory", required=True, type=str)
    args = parser.parse_args()
    direcotry = args.data_directory
    train_images = f"./{direcotry}/train_images.ubyte"
    train_labels = f"./{direcotry}/train_labels.ubyte"
    test_images = f"./{direcotry}/test_images.ubyte"
    test_labels = f"./{direcotry}/test_labels.ubyte"
    sample_count = 10
    shuffle = True

    train_gen = ImageBatchGenerator(train_images, train_labels, sample_count, shuffle)

    train_x, train_y = train_gen.read_input_raw(0)
    train_x, train_y = train_gen.reshape_batch_data(train_x, train_y)
    train_x, train_y = prepare_data_for_torch(np.array(train_x), np.array(train_y))
    net = LeNetTorchWrapper(epochs=200, steps_per_epoch=10, validation_steps=10, use_gpu=False, batch_size=1)
    net.load('cnn_torch20220624_160820.pt')
    net.model.eval()
    result = net.predict(np.array(train_x))
    print(result)


if __name__ == "__main__":
    main()
