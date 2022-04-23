# --coding:utf-8--
import torch
import torch.nn as nn
import numpy as np
from sklearn import preprocessing


def read_data(error=0, unit=[0], is_train=True):
    """
    Args:p
        error (int): The index of error, 0 means normal data
        is_train (bool): Read train or test data
    Returns:
        units' data
    """

    if is_train:
        suffix = '.dat'
    else:
        suffix = '_te.dat'
    fi = './data/d{:02d}{}'.format(error, suffix)
    data = np.fromfile(fi, dtype=np.float32, sep='   ')

    if fi == './data/d00.dat':
        data = data.reshape(-1, 500).T
        data = data[:, unit]
    else:
        data = data.reshape(-1, 52)
        data = data[:, unit]
    # if not is_train:
    #     data = data[160: ]
    return data, np.ones(data.shape[0], np.float32) * error


def get_train_data(unit):
    train_data, _ = read_data(error=0, unit=unit, is_train=True)
    train_data = preprocessing.StandardScaler().fit_transform(train_data)
    return train_data


def get_test_data(unit):
    test_data = []
    for i in range(22):
        data, _ = read_data(error=i, unit=unit, is_train=False)
        test_data.append(data)
    test_data = np.concatenate(test_data)
    train_data, _ = read_data(error=0, unit=unit, is_train=True)
    scaler = preprocessing.StandardScaler().fit(train_data)
    return test_data
