"""Utils for language models."""

import re
import numpy as np
import torch
import json
from torch.utils.data import TensorDataset

# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)
# print(NUM_LETTERS) 80

def generate_shake():

    TRAIN_DATA_DIR = "./data/shakespeare/data/train/"
    TEST_DATA_DIR = "./data/shakespeare/data/test/"

    TRAIN_DATA_NAME = "all_data_niid_1_keep_20_train_9.json"
    TEST_DATA_NAME = "all_data_niid_1_keep_20_test_9.json"

    with open(TRAIN_DATA_DIR+TRAIN_DATA_NAME) as json_file:
        train_data = json.load(json_file)

    with open(TEST_DATA_DIR+TEST_DATA_NAME) as json_file:
        test_data = json.load(json_file)

    n_clients = len(train_data["users"])
    TRIAL_USER_NAME = train_data["users"][0:n_clients] 
    trainDataset_local = {}
    testDataset_local = {}
    
    dataTrain_len = []
    dataTest_len = []

    for i in range(n_clients):   

        client_user_name = TRIAL_USER_NAME[i]
        num_samples_train = len(train_data["user_data"][client_user_name]['x'])
        num_samples_test = len(test_data["user_data"][client_user_name]['x'])

        user_train_data = train_data["user_data"][client_user_name]
        user_test_data = test_data["user_data"][client_user_name]

        X_train_i, y_train_i = process_x(user_train_data['x']),  process_y(user_train_data['y'])
        X_test_i, y_test_i = process_x(user_test_data['x']),  process_y(user_test_data['y'])
        #print(y_test_i)

        if i == 0:
            trainData_X = X_train_i
            testData_X = X_test_i
            trainData_y = y_train_i
            testData_y = y_test_i
        else:
            trainData_X = np.concatenate((trainData_X, X_train_i))
            testData_X = np.concatenate((testData_X, X_test_i))
            trainData_y = np.concatenate((trainData_y, y_train_i))
            testData_y = np.concatenate((testData_y, y_test_i))

        trainDataset_local[i] = TensorDataset(torch.from_numpy(X_train_i).long(),\
                                             torch.from_numpy(y_train_i).long()) 
        testDataset_local[i] = TensorDataset(torch.from_numpy(X_test_i).long(),\
                                             torch.from_numpy(y_test_i).long()) 

        dataTrain_len.append(len(trainDataset_local[i]))
        dataTest_len.append(len(testDataset_local[i]))

    #print(trainData_X.shape)
    #print(trainData_y.shape)
    #unique_y = np.unique(trainData_y)
    data_stats = (dataTrain_len, dataTest_len)
    trainDataset =  TensorDataset(torch.from_numpy(trainData_X).long(),\
                                             torch.from_numpy(trainData_y).long())
    testDataset = TensorDataset(torch.from_numpy(testData_X).long(),\
                                             torch.from_numpy(testData_y).long())
    
    return trainDataset, testDataset, trainDataset_local, testDataset_local, data_stats


def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    '''returns a list of character indices
    Args:
        word: string
    
    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices

def process_x(raw_x_batch):
    # each word is of length 20 
    # for word in raw_x_batch:
    #     print(len(word))
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    #x_batch = np.array(x_batch).T
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return np.array(y_batch)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


if __name__ == "__main__":
    trainDataset, testDataset, trainDataset_local, testDataset_local, data_stats = generate_shake()
    train_stats = data_stats[0]
    for i in range(len(train_stats)):
        print(i, train_stats[i])