from argparse import ArgumentParser

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
        self.conv1 = nn.Conv2d(1, 1, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(1, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNetTorchWrapper:
    """ Based on: https://www.kaggle.com/code/usingtc/lenet-with-pytorch/script """

    def __init__(self, epochs, steps_per_epoch, validation_steps, img_w=28, img_h=28, use_gpu=True):
        super().__init__()
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.model = LeNetTorch(img_w, img_h)
        self.use_gpu = use_gpu

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)

        if self.use_gpu:
            self.model = self.model.cuda()

    def build(self, train, test):

        return self.train(train, test)

    def train(self, train_data, labels, epoch_info_freq=1):
        train_size = train_data.shape[0]
        sample_index = 0
        batch_size = 100

        if isinstance(train_data, np.ndarray):
            train_data = torch.from_numpy(train_data)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        for epoch in range(self.epochs):
            if sample_index + batch_size >= train_size:
                sample_index = 0
            else:
                sample_index = sample_index + batch_size

            mini_data = Variable(train_data[sample_index:(sample_index + batch_size)].clone())
            mini_label = Variable(labels[sample_index:(sample_index + batch_size)].clone(), requires_grad=False)
            mini_data = mini_data.type(torch.FloatTensor)
            mini_label = mini_label.type(torch.LongTensor)
            if self.use_gpu:
                mini_data = mini_data.cuda()
                mini_label = mini_label.cuda()
            self.optimizer.zero_grad()
            mini_out = self.model(mini_data)
            mini_label = mini_label.view(batch_size)
            mini_loss = self.criterion(mini_out, mini_label)
            mini_loss.backward()
            self.optimizer.step()

            if (epoch + 1) % epoch_info_freq == 0:
                print(f"Epoch = {epoch+1}, Loss = {mini_loss.item()}")

    def save(self, path):
        raise NotImplementedError
        # self.model.save(path)

    def load(self, path):
        raise NotImplementedError

    def predict(self, test_input):
        raise NotImplementedError
        # return self.model.predict(test_input)


def prepare_data_for_torch(train_x: numpy.ndarray, train_y: numpy.ndarray):
    return np.array(train_x).reshape(100000, 1, 28, 28), \
        np.array(train_y).reshape(100000, 1)

    pass


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_directory", required=True, type=str)
    args = parser.parse_args()
    direcotry = args.data_directory
    train_images = f"./{direcotry}/train_images.ubyte"
    train_labels = f"./{direcotry}/train_labels.ubyte"
    # test_images = f"./{direcotry}/test_images.ubyte"
    # test_labels = f"./{direcotry}/test_labels.ubyte"
    sample_count = 100000
    shuffle = True

    train_gen = ImageBatchGenerator(train_images, train_labels, sample_count, shuffle)

    train_x, train_y = train_gen.read_input(0)
    train_x, train_y = train_gen.reshape_batch_data(train_x, train_y)
    train_x, train_y = prepare_data_for_torch(train_x, train_y)
    net = LeNetTorchWrapper(epochs=100, steps_per_epoch=10, validation_steps=10, use_gpu=False)
    net.train(np.array(train_x), np.array(train_y))


if __name__ == "__main__":
    main()
