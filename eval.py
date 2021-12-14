import keras
import sys
import h5py
import numpy as np

# clean_data_filename = str(sys.argv[1])
# poisoned_data_filename = str(sys.argv[2])
# model_filename = str(sys.argv[3])
clean_data_filename = '/home/ming/PycharmProjects/CSAW-HackML-2020/lab3/data/cl/valid.h5'
poisoned_data_filename = '/home/ming/PycharmProjects/CSAW-HackML-2020/lab3/data/bd/bd_valid.h5'
model_filename = '/home/ming/PycharmProjects/CSAW-HackML-2020/lab3/models/bd_net.h5'

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def main(bd_model=None):
    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)

    if not bd_model:
        bd_model = keras.models.load_model(model_filename)

    cl_label_p = np.argmax(bd_model.predict(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test))*100
    print('Clean Classification accuracy:', clean_accuracy)
    
    bd_label_p = np.argmax(bd_model.predict(bd_x_test), axis=1)
    asr = np.mean(np.equal(bd_label_p, bd_y_test))*100
    print('Attack Success Rate:', asr)
    return (clean_accuracy, asr)

if __name__ == '__main__':
    main()
