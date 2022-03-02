import os
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


def plot_accuracy(history, batch_size=None, shuffle=None, reps=None, show=False, fname=None):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    if show:
        plt.show()

    plt.title(f'Accuracy, {reps} repetitions, batch size: {batch_size}, shuffle {bool(shuffle)}')
    plt.savefig(f'{fname}_{batch_size}_{reps}_{shuffle}.png')
