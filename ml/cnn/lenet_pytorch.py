import random
from argparse import ArgumentParser
from datetime import datetime

import numpy
import numpy as np
import torch

from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from ml.image_batch_generator import ImageBatchGenerator


class LeNetTorch(nn.Module):
    """ Based on: https://www.kaggle.com/code/usingtc/lenet-with-pytorch/script """

    def __init__(self, img_width, img_height):
        super(LeNetTorch, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # print(f"c1.shape = {x.shape}")
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # print(f"c2.shape = {x.shape}")
        x = x.view(-1, self.num_flat_features(x))
        # print(f"view.shape = {x.shape}")
        x = F.relu(self.fc1(x))
        # print(f"fc1.shape = {x.shape}")
        x = F.relu(self.fc2(x))
        # print(f"fc2.shape = {x.shape}")
        x = self.fc3(x)
        # print(f"fc3.shape = {x.shape}")
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNetTorchWrapper:
    """ Based on: https://www.kaggle.com/code/usingtc/lenet-with-pytorch/script """

    def __init__(self, epochs, steps_per_epoch, validation_steps, batch_size=100, img_w=28, img_h=28, use_gpu=True):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size

        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        self.model = LeNetTorch(img_w, img_h)
        self.img_w = img_w
        self.img_h = img_h

        self.use_gpu = use_gpu

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)

        if self.use_gpu:
            self.model = self.model.cuda()

    def build(self, train, test):
        return self.train(train, test)

    def train(self, train_data: np.ndarray, labels: np.ndarray, epoch_info_freq: int = 1):
        train_size = train_data.shape[0]
        sample_index = 0

        if isinstance(train_data, np.ndarray):
            train_data = torch.from_numpy(train_data)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        for epoch in range(self.epochs):
            if sample_index + self.batch_size >= train_size:
                sample_index = 0
            else:
                sample_index = sample_index + self.batch_size

            mini_data = Variable(train_data[sample_index:(sample_index + self.batch_size)].clone())
            mini_label = Variable(labels[sample_index:(sample_index + self.batch_size)].clone(), requires_grad=False)
            mini_data = mini_data.type(torch.FloatTensor)
            mini_label = mini_label.type(torch.LongTensor)
            if self.use_gpu:
                mini_data = mini_data.cuda()
                mini_label = mini_label.cuda()
            self.optimizer.zero_grad()
            mini_out = self.model(mini_data)
            mini_label = mini_label.view(self.batch_size)
            mini_loss = self.criterion(mini_out, mini_label)
            mini_loss.backward()
            self.optimizer.step()

            if (epoch + 1) % epoch_info_freq == 0:
                print(f"Epoch = {epoch + 1}, Loss = {mini_loss.item()}")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def save_onnx(self, path: str, opset: int = 13):
        model_input = torch.randn(self.batch_size, 1, self.img_h, self.img_w, requires_grad=True)
        input_names = ['act']
        output_names = ['out']
        torch.onnx.export(self.model, model_input, path, input_names=input_names, output_names=output_names,
                          opset_version=opset)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def predict(self, test_input: np.ndarray):
        test_input = torch.from_numpy(test_input)
        mini_data = Variable(test_input.clone())
        mini_data = mini_data.type(torch.FloatTensor)
        if self.use_gpu:
            mini_data = mini_data.cuda()
        self.optimizer.zero_grad()
        mini_out = self.model(mini_data)

        return mini_out


def prepare_data_for_torch(train_x: numpy.ndarray, train_y: numpy.ndarray):
    return np.array(train_x).reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]), \
           np.array(train_y).reshape(train_y.shape[0], 1)


def pick_2n_test_images(n, images, labels):
    ones_list = [i for i, value in enumerate(labels) if value == 1]
    zeros_list = [i for i, value in enumerate(labels) if value == 0]
    x_set = []
    y_set = [random.sample(labels(ones_list), n), random.sample(labels(zeros_list), n)]
    x_set.append(random.sample(labels(ones_list), n))
    x_set.append(random.sample(labels(zeros_list), n))
    return x_set, y_set


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_directory", required=True, type=str)
    args = parser.parse_args()
    direcotry = args.data_directory
    train_images = f"./{direcotry}/train_images.ubyte"
    train_labels = f"./{direcotry}/train_labels.ubyte"
    sample_count = 100000
    shuffle = True
    dataset_mean = 0.258
    dataset_std = 0.437

    train_gen = ImageBatchGenerator(train_images, train_labels, sample_count, shuffle)

    train_x, train_y = train_gen.read_input_raw(0)
    train_x, train_y = train_gen.reshape_batch_data(train_x, train_y)
    train_x, train_y = prepare_data_for_torch(np.array(train_x), np.array(train_y))
    train_x = normalize_training_data(dataset_mean, dataset_std, train_x)
    net = LeNetTorchWrapper(epochs=1000, steps_per_epoch=10, validation_steps=10, use_gpu=False)
    net.train(np.array(train_x), np.array(train_y))

    date_str = datetime.today().strftime("%Y%m%d_%H%M%S")
    save_path = rf'models/cnn_torch_{date_str}'
    print(f"Saving model to: {save_path}.pt and .onnx")

    net.save(f"{save_path}.pt")
    net.save_onnx(f"{save_path}.onnx", opset=10)


def normalize_training_data(dataset_mean, dataset_std, train_x):
    train_x = train_x.astype(np.float32) / 255
    train_x = (train_x - dataset_mean) / dataset_std
    return train_x


if __name__ == "__main__":
    main()
