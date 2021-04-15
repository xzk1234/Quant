import numpy as np
import pandas as pd
import os
import shutil
import time
import datetime
import sys


def generate_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        shutil.rmtree(folder_name)
        os.makedirs(folder_name)


def read_and_dropna(data_dir='./stock_data/000001.XSHE.csv'):
    data = pd.read_csv(data_dir, encoding='gbk')
    data = data.replace(np.float64(0.0), np.nan)
    data = data.dropna(axis=1)
    columns = data.columns
    data = np.array(data)
    return data, columns


def diff_op(data):
    row, col = data.shape
    new_data = np.zeros((row, col))
    for i in range(row - 1, 0, -1):
        if i != 0:
            new_data[i] = data[i] - data[i - 1]
    return new_data


def return_op(data):
    row, col = data.shape
    new_data = np.zeros((row, col))
    for i in range(row - 1, -1, -1):
        if i != 0:
            if 0 in data[i - 1]:
                print(i)
            else:
                new_data[i] = data[i] / data[i - 1]
        else:
            new_data[i] = 1
    return new_data


def generate_label(data, label_num=2, mode='diff', threshold1=None, threshold2=None, normalized=False, mean=None, std=None):
    row, col = data.shape
    new_data = np.zeros((row, col + label_num))
    new_data[:, :col] = data[:, :col]
    if normalized:
        data[:, -1] = data[:, -1] * std + mean
    if mode == 'diff':
        judge = 0
    else:
        judge = 1
    if label_num == 2:
        for i in range(row):
            if data[i, -1] > judge:
                new_data[i, -2] = 1
            else:
                new_data[i, -1] = 1
    else:
        for i in range(row):
            if data[i, -1] > threshold1:
                new_data[i, -3] = 1
            elif data[i, -1] < threshold2:
                new_data[i, -2] = 1
            else:
                new_data[i, -1] = 1
    return new_data


def generate_second_order_data(data_dir):
    data_name = data_dir.split('/')[-1].split('.')[0]
    data = pd.read_csv(data_dir)
    data = np.array(data)
    n_feature = data.shape[1] - 1
    second_order_data = np.zeros((len(data), n_feature * n_feature + 1))
    second_order_data[:, -1] = data[:, -1]
    for i in range(n_feature):
        feature1 = data[:, i]
        for j in range(n_feature):
            start = time.time()
            feature2 = data[:, j]
            for k in range(len(data)):
                if feature1[k] > 1:
                    second_order_data[k, i * n_feature + j] = feature1[k] ** feature2[k]
                else:
                    second_order_data[k, i * n_feature + j] = feature1[k] ** (1 / feature2[k])
            end = time.time()
            print('Corr ({}, {}), time consuming: {}'.format(i, j, end - start))
    np.save('./stock_data/' + data_name + '_second_order.npy', second_order_data)


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""
    def __init__(self, fn, model_dir='./logs/'):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        logdir = model_dir + fn
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        if len(os.listdir(logdir)) != 0:
            ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                            "Will you proceed [y/N]? ")
            if ans in ['y', 'Y']:
                shutil.rmtree(logdir)
            else:
                exit(1)
        self.set_dir(logdir)

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()


