import json
import logging
import os

import numpy as np

from ml.common import byte_arr_to_int_arr


def try_create_directory(directory_path):
    if not os.path.exists(directory_path):
        logging.info(f"Creating directory: {directory_path}")
        os.mkdir(directory_path)


def load_json_config(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def read_data_from_single_file(fname, count=None, element_size=784):
    data = []
    stop_flag = False
    idx = 0

    with open(fname, 'rb') as f:
        while not stop_flag and (count is None or idx < count):
            img_data = f.read(element_size)
            if len(img_data) == element_size:
                data.append(byte_arr_to_int_arr(img_data))
                idx += 1
            else:
                stop_flag = True
    logging.info(f"{fname} loaded. count: {idx} imgs")
    return np.array(data)


def load_data(data_dir, fname_x="test_images.ubyte", fname_y="test_labels.ubyte", size_x=784, size_y=1, count=None):
    path_x = f"{data_dir}/{fname_x}"
    path_y = f"{data_dir}/{fname_y}"

    data = read_data_from_single_file(path_x, count, element_size=size_x)
    labels = read_data_from_single_file(path_y, count, element_size=size_y)

    return data, labels
