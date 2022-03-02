import argparse

from ml.common import plot_accuracy
from story import Story, Stories
from train_cnn import read_json


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--json", required=True,
                    help="path to the JSON input file")

    args = vars(ap.parse_args())
    return args


def main():
    args = parse_args()
    history = read_json(args["json"])
    batch_sizes = [500, 1000, 2000, 4000, 8000]
    # create_plots(batch_sizes, history)
    create_mean_plots(batch_sizes, history)


def create_plots(batch_sizes, history):
    repetitions = 1
    idx = 0
    shuffle = 0
    file_number = 0
    for shuffle_elt in history:
        for batch_size_elt in shuffle_elt:
            for rep_elt in batch_size_elt:
                history_part = Story(batch_size=batch_sizes[idx], shuffle=shuffle, reps=repetitions, history=rep_elt)
                plot_accuracy(history_part, history_part.batch_size, history_part.shuffle, history_part.reps, False,
                              f'plots/plot_{file_number}_')
                repetitions += 1
                file_number += 1
            idx += 1
            repetitions = 0
        idx = 0
        shuffle += 1


def create_mean_plots(batch_sizes, history):
    idx = 0
    shuffle = 0
    file_number = 0
    for shuffle_elt in history:
        for batch_size_elt in shuffle_elt:
            histories = [rep_elt for rep_elt in batch_size_elt]
            history_part = Stories(batch_size=batch_sizes[idx], shuffle=shuffle, reps=1, histories=histories)
            plot_accuracy(history_part, history_part.batch_size, history_part.shuffle, history_part.reps, False,
                          f'plots/plot_{file_number}_')
            file_number += 1
            idx += 1
        idx = 0
        shuffle += 1


if __name__ == "__main__":
    main()
