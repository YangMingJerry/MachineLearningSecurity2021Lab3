import sys
import os

import keras
from keras import backend as K
from tfkerassurgeon.operations import Surgeon, delete_channels
import h5py
import numpy as np

from eval import main as evaluate


def get_activation(model, layer_num, X):
    layer = model.layers[layer_num]
    input_tensor = model.input
    # output tensor of the given layer
    layer_output = layer.output
    # get the output with respect to the input
    func = K.function([input_tensor], [layer_output])
    # get activation for the test image
    activation = np.mean(func(X), axis=(0, 1, 2, 3))
    return activation

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))
    return x_data, y_data

model_filename = '/home/ming/PycharmProjects/CSAW-HackML-2020/lab3/models/bd_net.h5'
clean_data_filename = '/home/ming/PycharmProjects/CSAW-HackML-2020/lab3/data/cl/test.h5'
cl_x_test, cl_y_test = data_loader(clean_data_filename)
model = keras.models.load_model(model_filename)
activation = get_activation(model, 5, cl_x_test)
seq_sort = np.argsort(activation)
i = 0
clean_acc_start,_ = evaluate(model)
while 1:
    channel = seq_sort[i]
    # model = delete_channels(model, model.layers[5], [7,26,22])
    weights = model.layers[5].weights[0][:, :, :, channel]
    i += 1
    clean_acc,_ = evaluate(model)
    if clean_acc <= clean_acc_start - 2.:
        break
print('debug')


