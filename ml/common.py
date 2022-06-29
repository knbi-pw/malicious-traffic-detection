import os
from datetime import datetime

import cv2
from matplotlib import pyplot as plt


def int_list_from_bytes(_bytes: bytes):
    return [b for b in _bytes]


def display_resized_image(fname, image, factor=20):
    img_display = cv2.resize(image, None, None, factor, factor, interpolation=cv2.INTER_AREA)
    cv2.imshow(fname, img_display)
    cv2.waitKey(1)


def byte_arr_to_int_arr(bytearr):
    return list(bytearr)


def read_file_size(fname):
    return os.path.getsize(fname)


def plot_accuracy(history, batch_size=None, shuffle=None, reps=None, show=False, fname=None, display_title=False):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.grid()
    if show:
        plt.show()

    fig_name = f'{fname}_{batch_size}_{reps}_{shuffle}.png'
    print(f"Plotting to {fig_name}")

    if display_title is False:
        title = ""
    else:
        title = "Accuracy"
        if reps:
            title += f", {reps} repetitions"
        if batch_size:
            title += f", batch size: {batch_size}"
        if shuffle:
            title += f", shuffle: {bool(batch_size)}"

    plt.title(title)
    plt.savefig(fig_name)
