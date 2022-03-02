import argparse
import logging
import json
from ml.array_batch_generator import ArrayBatchGenerator
from ml.common import plot_accuracy
from ml.dl.deep_in_the_dark import DeepModel
import tensorflow as tf


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("-j", "--json_config_path", required=True,
                    help="path to the JSON config file")
    args = vars(ap.parse_args())
    return args


def read_json(jsonPath):
    with open(jsonPath) as f:
        data = json.load(f)
    return data


def main():
    logging.basicConfig(level=logging.DEBUG)

    argument = parse_args()
    config = read_json(argument["json_config_path"])

    model = DeepModel(config["epochs"], config["steps_per_epoch"], config["validation_steps"])
    batch_size = config["batch_size"]
    train_dataset = ArrayBatchGenerator(config["train_images"], config["train_labels"], batch_size, config["shuffle"])
    validation_dataset = ArrayBatchGenerator(config["test_images"], config["test_labels"], batch_size,
                                             config["shuffle"])
    model, history = model.model_build(train_dataset, validation_dataset)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    if config["model"] is not None:
        model.save(config["model"])
    if config["plot"] == 1:
        logging.getLogger().setLevel(logging.DEBUG)
        plot_accuracy(history)


if __name__ == "__main__":
    main()
