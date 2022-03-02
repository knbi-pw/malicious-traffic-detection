import json
import logging

from ml.common import read_file_size
from ml.array_batch_generator import ArrayBatchGenerator
from ml.dl.deep_in_the_dark import DeepModel


def main():
    logging.basicConfig(level=logging.DEBUG)

    train_images = "preprocess_result/train_images.ubyte"
    train_labels = "preprocess_result/train_labels.ubyte"
    test_images = "preprocess_result/test_images.ubyte"
    test_labels = "preprocess_result/test_labels.ubyte"

    histories = []
    batch_sizes = [500, 1000, 2000, 4000, 8000]
    epochs = 10
    repetitions = 10
    for shuffle in [0, 1]:
        histories_per_batch_size = []
        for batch_size in batch_sizes:
            steps_per_epoch, validation_steps = get_steps(batch_size, test_labels, train_labels)
            history_reps = []
            for rep in range(repetitions):
                logging.info(f"Running Model({epochs},{steps_per_epoch},{validation_steps}), rep:{rep}")
                history = train_model(batch_size, epochs, shuffle, steps_per_epoch, validation_steps,
                                      test_labels, train_images, train_labels, test_images)
                history_reps.append(history.history)
            histories_per_batch_size.append(history_reps)
        histories.append(histories_per_batch_size)
    with open('train_deep_multiple_result.json', 'w') as f:
        json.dump(histories, f)


def get_steps(batch_size, test_labels, train_labels):
    steps_per_epoch = read_file_size(train_labels) // batch_size
    validation_steps = read_file_size(test_labels) // batch_size
    return steps_per_epoch, validation_steps


def train_model(batch_size, epochs, shuffle, steps_per_epoch, validation_steps, test_labels, train_images, train_labels,
                test_images):
    model = DeepModel(epochs, steps_per_epoch, validation_steps)
    train_dataset = ArrayBatchGenerator(train_images, train_labels, batch_size, shuffle)
    validation_dataset = ArrayBatchGenerator(test_images, test_labels, batch_size, shuffle)
    model, history = model.model_build(train_dataset, validation_dataset)
    return history


if __name__ == "__main__":
    main()
