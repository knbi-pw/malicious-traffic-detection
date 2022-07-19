from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--count", type=int, default=None)

    return parser.parse_args()


def save_png(output_dir, idx, image: np.array):
    fname = f"{output_dir}/image_{idx}.png"
    image_png = Image.fromarray(image)
    image_png.save(fname)


def main():
    args = parse_args()
    img_width = args.image_width
    output_dir = args.output_dir
    img_size = img_width**2
    count = args.count
    idx = 0

    print("Begin conversion...")
    with open(args.input_file, "rb") as f:
        if count is not None:
            pbar = tqdm(total=count)
        while count is None or idx < count:
            f.seek(idx*img_size)
            image = f.read(img_size)

            if len(image) >= 0:
                image = np.array(list(image))  # bytes to int arr to np
                image = image.reshape(img_width, img_width)
                save_png(output_dir, idx, image)
                idx += 1

                if count is not None:
                    pbar.update(1)
            else:
                break

    print("Done!")


if __name__ == "__main__":
    main()
