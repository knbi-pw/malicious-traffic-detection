import logging
import numpy as np
from tensorflow import keras

from ml.common import byte_arr_to_int_arr, read_file_size


def shuffle_dataset_in_unision(x, y):
    shuffler = np.random.permutation(len(x))
    x = np.array(x)[shuffler]
    y = np.array(y)[shuffler]
    return x, y


class ImageBatchGenerator(keras.utils.Sequence):
    def __init__(self, images_fname, labels_fname, batch_size=2000, shuffle=False, image_width=28):
        self.batch_size = batch_size
        self.images_fname = images_fname
        self.labels_fname = labels_fname

        self.shuffle = shuffle

        self.image_width = image_width
        self.image_size = self.image_width * self.image_width

        self.fsize_x = read_file_size(self.images_fname)
        self.fsize_y = read_file_size(self.labels_fname)
        self.raise_when_inconsistent_lengths()

        self.batches_count = self.fsize_x // (self.batch_size * self.image_size)
        logging.debug(f"[BatchGenerator __init__] Batches count: {self.batches_count}")

    def raise_when_inconsistent_lengths(self):
        if self.fsize_x != self.fsize_y * self.image_size:
            raise Exception(f"Number of images differs from number of labels in batch generator input data: "
                            f"{self.fsize_x} != {self.fsize_y}*{self.image_size}")

    def reshape_batch_data(self, data, values):
        img_data = [np.reshape(f_data, (self.image_width, self.image_width, 1)) for f_data in data]
        y_data = [g_data[0] for g_data in values]
        return img_data, y_data

    def __len__(self):
        return self.batches_count

    def __getitem__(self, idx):
        logging.debug(f"Starting __getitem__(self, {idx})")
        batch_x, batch_y = self.read_input(idx)
        batch_x, batch_y = self.reshape_batch_data(batch_x, batch_y)

        if self.shuffle:
            batch_x, batch_y = shuffle_dataset_in_unision(batch_x, batch_y)

        logging.debug(f"[BatchGenerator __getitem__] Batch_x shape: {len(batch_x[0])}")
        batch_x = keras.utils.normalize(batch_x)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        logging.debug(f"batch_x.shape = {batch_x.shape}, batch_y = {batch_y.shape}")

        return batch_x, batch_y

    def read_input(self, idx):
        x_data = []
        y_data = []

        logging.debug(f"[BatchGenerator __read_input__] batch_size: {self.batch_size}")
        start_idx = idx * self.batch_size
        current_idx = start_idx
        end_idx = start_idx + self.batch_size

        with open(self.images_fname, 'rb') as f, open(self.labels_fname, 'rb') as g:
            f.seek(start_idx * self.image_size)
            g.seek(start_idx)
            while current_idx * self.image_size < end_idx * self.image_size:
                img_data_x = f.read(self.image_size)
                img_data_y = g.read(1)
                if len(img_data_x) == self.image_size and len(img_data_y) == 1:
                    x_data.append(byte_arr_to_int_arr(img_data_x))
                    y_data.append(byte_arr_to_int_arr(img_data_y))
                    current_idx += 1
                else:
                    break
        logging.debug(f"[BatchGenerator __read_input__] read {len(x_data[0])} bytes successfully")
        return x_data, y_data