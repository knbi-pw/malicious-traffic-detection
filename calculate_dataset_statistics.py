from ml.image_batch_generator import ImageBatchGenerator
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--labels", required=True, type=str)
    args = parser.parse_args()

    data = args.data
    labels = args.labels

    sample_count = 10
    shuffle = False

    gen = ImageBatchGenerator(data, labels, sample_count, shuffle)
    x, y = gen.read_input_numpy(idx=0)
    print(f"mean: {x.mean()}, stdev: {x.std()}")


if __name__ == "__main__":
    main()
