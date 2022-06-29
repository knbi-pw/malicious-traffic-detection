import argparse
import logging
import json
from enum import Enum

import ml.common
from ml.cnn.novel_cnn import NovelCnnModel
from ml.image_batch_generator import ImageBatchGenerator
from ml.cnn.lenet import LeNetModel


class ModelType(Enum):
    lenet = LeNetModel
    novel = NovelCnnModel


def parse_model_type(model_type: str):
    if model_type == "lenet":
        return ModelType.lenet
    elif model_type == "novel":
        return ModelType.novel
    raise Exception("Incorrect input model type")


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("-j", "--json_config_path", required=True,
                    help="path to the JSON config file")
    ap.add_argument("--model_type", required=False, default=ModelType.lenet)
    args = vars(ap.parse_args())
    args["model_type"] = parse_model_type(args["model_type"])
    return args


def read_json(jsonPath):
    with open(jsonPath) as f:
        data = json.load(f)
    return data


def main():
    logging.basicConfig(level=logging.WARNING)

    argument = parse_args()
    config = read_json(argument["json_config_path"])

    model = argument["model_type"].value(config["epochs"], config["steps_per_epoch"], config["validation_steps"])
    batch_size = config["batch_size"]
    train_gen = ImageBatchGenerator(config["train_images"], config["train_labels"], batch_size, config["shuffle"])
    test_gen = ImageBatchGenerator(config["test_images"], config["test_labels"], batch_size, config["shuffle"])
    history = model.model_build(train_gen, test_gen)

    if config["model"] is not None:
        model.model_save(config["model"])
    if config["plot"] == 1:
        logging.getLogger().setLevel(logging.WARNING)
        ml.common.plot_accuracy(history)


if __name__ == "__main__":
    main()
