import numpy as np
import random
from utils import *
import pandas as pd
import time
import sys


class Input_data:
    def __init__(self, batch_size, n_step_encoder, n_step_decoder, n_hidden_encoder, data, mode='return'):
        # read the data
        self.data = data
        self.n_feature = self.data.shape[1] - 1
        self.n_train = int(len(self.data) * 0.7)
        self.n_val = int(len(self.data) * 0.1)
        self.n_test = len(self.data) - self.n_train - self.n_val
        self.train = self.data[:self.n_train, :]
        self.val = self.data[self.n_train:self.n_train + self.n_val, :]
        self.test = self.data[self.n_train + self.n_val:, :]

        # parameters for the network
        self.batch_size = batch_size
        self.n_hidden_state = n_hidden_encoder
        self.n_step_encoder = n_step_encoder
        self.n_step_decoder = n_step_decoder

        self.n_label = 2

        # data normalization
        self.mean = np.mean(self.train, axis=0)
        self.stdev = np.std(self.train, axis=0)
        # in case the stdev=0,then we will get nan
        for i in range(len(self.stdev)):
            if self.stdev[i] < 0.00000001:
                self.stdev[i] = 1
        self.train = (self.train - self.mean) / self.stdev
        self.test = (self.test - self.mean) / self.stdev
        self.val = (self.val - self.mean) / self.stdev
        self.train = generate_label(self.train, 3, mode='return', normalized=True, mean=self.mean[-1],
                                    std=self.stdev[-1], threshold1=1.000001, threshold2=0.999999)
        self.test = generate_label(self.test, 3, mode='return', normalized=True, mean=self.mean[-1],
                                   std=self.stdev[-1], threshold1=1.000001, threshold2=0.999999)
        self.val = generate_label(self.val, 3, mode='return', normalized=True, mean=self.mean[-1],
                                  std=self.stdev[-1], threshold1=1.000001, threshold2=0.999999)
        self.train_index = self.generate_index(self.train)
        self.val_index = self.generate_index(self.val)
        self.test_index = self.generate_index(self.test)
        # self.train_x, self.train_label, self.train_prev_y = self.generate_dataset(self.train)
        # self.val_x, self.val_label, self.val_prev_y = self.generate_dataset(self.val)
        # self.test_x, self.test_label, self.test_prev_y = self.generate_dataset(self.test)

    def generate_index(self, data):
        index = []
        for i in range(len(data) - self.n_step_decoder):
            if data[i + self.n_step_decoder, -1] != 1:
                index.append(i)
        return index

    def generate_dataset(self, data):
        count = 0
        for i in range(len(data) - self.n_step_decoder):
            if data[i + self.n_step_decoder, -1] != 1:
                count += 1
        dataset_x = []
        dataset_label = []
        dataset_prev_y = []
        start = time.time()
        for i in range(len(data) - self.n_step_decoder):
            if i % 10000 == 0:
                end = time.time()
                print('The {}th data, time consuming: {}'.format(i, end - start))
                start = time.time()
            if data[i + self.n_step_decoder, -3] == 1:
                dataset_x.append(data[i:i + self.n_step_encoder, :-4])
                dataset_label.append([1, 0])
                dataset_prev_y.append(data[i:i + self.n_step_decoder, -4])
            elif data[i + self.n_step_decoder, -2] == 1:
                dataset_x.append(data[i:i + self.n_step_encoder, :-4])
                dataset_label.append([0, 1])
                dataset_prev_y.append(data[i:i + self.n_step_decoder, -4])
            else:
                continue
        print('\n')
        return np.array(dataset_x), np.array(dataset_label), np.expand_dims(np.array(dataset_prev_y), -1)

    def get_sample(self, index, data):
        batch_size = len(index)
        x = np.zeros((batch_size, self.n_step_encoder, self.n_feature))
        label = np.zeros((batch_size, self.n_label))
        prev_y = np.zeros((batch_size, self.n_step_decoder, 1))
        for i in range(batch_size):
            x[i] = data[index[i]:index[i] + self.n_step_encoder, :-4]
            prev_y[i] = np.expand_dims(data[index[i]:index[i] + self.n_step_decoder, -4], -1)
            if data[index[i] + self.n_step_decoder, -3] == 1:
                label[i] = np.array([1, 0])
            elif data[index[i] + self.n_step_decoder, -2] == 1:
                label[i] = np.array([0, 1])
            else:
                print('Label Error!')
                sys.exit(0)
        encoder_states = np.swapaxes(x, 1, 2)
        return x, label, prev_y, encoder_states

    def next_batch(self):
        # generate of a random index from the range [0, self.n_train -self.n_step_decoder +1]
        index_index = random.sample(list(np.arange(0, len(self.train_index))), self.batch_size)
        index = []
        for i in range(len(index_index)):
            index.append(self.train_index[index_index[i]])
        batch_x, batch_label, batch_prev_y, encoder_states = self.get_sample(index, self.train)
        return batch_x, batch_label, batch_prev_y, encoder_states

    def training(self):
        train_x, train_label, train_prev_y, train_encoder_states = self.get_sample(self.train_index, self.train)
        return train_x, train_label, train_prev_y, train_encoder_states

    def validation(self):
        val_x, val_label, val_prev_y, val_encoder_states = self.get_sample(self.val_index, self.val)
        return val_x, val_label, val_prev_y, val_encoder_states

    def testing(self):
        test_x, test_label, test_prev_y, test_encoder_states = self.get_sample(self.test_index, self.test)
        return test_x, test_label, test_prev_y, test_encoder_states