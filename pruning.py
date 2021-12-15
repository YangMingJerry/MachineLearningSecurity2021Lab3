import sys
import os

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from kerassurgeon.operations import Surgeon, delete_channels
import h5py
import numpy as np
import matplotlib.pyplot as plt

from eval import main as evaluate

def model_save(model,info,clean_acc, asr):
    name = f'pruned_bd_net_by_{info}_acc={clean_acc}_asr={asr}.h5'
    path = os.path.join('E:\pycharmProjects\MachineLearningSecurity2021Lab3\models',name)
    model.save(path)
    return

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


model_filename = 'E:\pycharmProjects\CSAW-HackML-2020\lab3\models\\bd_net.h5'
# clean_data_filename = '/home/ming/PycharmProjects/CSAW-HackML-2020/lab3/data/cl/test.h5'
clean_data_filename = 'E:\pycharmProjects\CSAW-HackML-2020\lab3\data\cl\\valid.h5'
cl_x_test, cl_y_test = data_loader(clean_data_filename)
model = keras.models.load_model(model_filename)
activation = get_activation(model, 5, cl_x_test)
seq_sort = np.argsort(activation)
i = 0
clean_acc_start,_ = evaluate(model)
accs = []
asrs = []
inds = []
save_2, save_4, save_10 = 0,0,0
while i<59:
    channel = seq_sort[i]
    print(f'Deleting {i+1}th/60 channels: {channel}')
    # new_model = delete_channels(model, model.layers[6], channels)
    # weights = model.layers[5].weights[0][:, :, :, channel]
    w = model.get_weights()
    w[5][channel] = 0.
    model.set_weights(w)
    w_l = model.layers[5].get_weights()
    w_l[0][:, :, :, channel] = 0.
    model.layers[5].set_weights(w_l)

    i += 1
    clean_acc, asr = evaluate(model)
    if clean_acc <= clean_acc_start - 2 and not save_2:
        model_save(model,'2',clean_acc,asr)
        save_2 = 1
    if clean_acc <= clean_acc_start - 4 and not save_4:
        model_save(model,'4',clean_acc, asr)
        save_4 = 1
    if clean_acc <= clean_acc_start - 10 and not save_10:
        model_save(model,'10',clean_acc, asr)
        save_10 = 1

    accs.append(clean_acc)
    asrs.append(asr)
    inds.append(i)

plt.figure()
plt.plot(inds,accs,'g',inds,asrs,'r')
plt.show()
print('debug')

def bd_net_prune():
    pass
