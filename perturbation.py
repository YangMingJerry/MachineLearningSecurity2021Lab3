import tensorflow.keras as keras
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from eval import data_loader

def eval():
    threshold = 0.999999
    labels = bd_model.predict(cl_x_test)
    labels_raw = bd_model.predict(cl_x_test_raw)
    cl_label_p = np.argmax(labels_raw, axis=1)
    for i,label in enumerate(labels):
        if max(label) > threshold:
            cl_label_p[i] = 1283
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test)) * 100
    print('Clean Classification accuracy:', clean_accuracy)


    labels = bd_model.predict(bd_x_test)
    labels_raw = bd_model.predict(bd_x_test_raw)
    bd_label_p = np.argmax(labels_raw, axis=1)
    for i,label in enumerate(labels):
        if max(label) > threshold:
            bd_label_p[i] = 1283
    asr = np.mean(np.equal(bd_label_p, bd_y_test)) * 100
    print('Attack Success Rate:', asr)

if __name__ == '__main__':
    clean_data_filename = 'E:\pycharmProjects\CSAW-HackML-2020\lab3\data\cl\\valid.h5'
    poisoned_data_filename = 'E:\pycharmProjects\CSAW-HackML-2020\lab3\data\\bd\\bd_valid.h5'
    path_bd = 'models/bd_net.h5'
    bd_model = keras.models.load_model(path_bd)
    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    cl_x_test_raw, _ = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)
    bd_x_test_raw, _ = data_loader(poisoned_data_filename)
    shift = cl_x_test[0]
    num_shift = 5
    for x in cl_x_test[1:num_shift-1]:
        shift += x
    shift = shift / num_shift
    for i, x in enumerate(cl_x_test):
        # cl_x_test[i] = x/num_shift + shift
        cl_x_test[i] = x + shift

    for i, x in enumerate(bd_x_test):
        # bd_x_test[i] = x/num_shift + shift
        bd_x_test[i] = x + shift
    y1 = bd_model.predict(cl_x_test)
    y2 = bd_model.predict(bd_x_test)
    print(min([v.max() for v in y1]))
    print(max([v.max() for v in y1]))
    print(min([v.max() for v in y2]))
    print(max([v.max() for v in y2]))
    data1 = [v.max() for v in y1]
    data2 = [v.max() for v in y2]
    kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)
    plt.figure()
    # Plot


    plt.hist(data1, **kwargs, color='r', label='Ideal')
    plt.hist(data2, **kwargs, color='g', label='Fair')
    plt.show()
    eval()
    print('debug')