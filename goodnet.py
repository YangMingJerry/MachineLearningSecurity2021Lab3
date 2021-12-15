import tensorflow.keras as keras
import numpy as np
from PIL import Image

from eval import data_loader
class GoodNet:
    def __init__(self, path_bd, path_pruned):
        self.model_b = keras.models.load_model(path_bd)
        self.model_p = keras.models.load_model(path_pruned)
        self.num_class = self.model_p.output.shape[1]

    def predict(self, x):
        y_b = np.argmax(self.model_b.predict(x), axis=1)
        y_p = np.argmax(self.model_p.predict(x), axis=1)
        for i in range(len(y_b)):
            if y_b[i] != y_p[i]:
                y_p[i] = self.num_class+1
        return y_p


def goodnet_eval():
    bd_x_test, bd_y_test = data_loader(path_valid_x_bd)
    cl_x_test, cl_y_test = data_loader(path_clean_x)
    goodnet = GoodNet(path_bd, path_pruned)
    cl_label_p = goodnet.predict(cl_x_test)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test))*100
    print('Clean Classification accuracy:', clean_accuracy)
    bd_label_p = goodnet.predict(bd_x_test)
    asr = np.mean(np.equal(bd_label_p, bd_y_test))*100
    print('Attack Success Rate:', asr)

class Eval:
    def __init__(self, path_bd, path_pruned):
        self.goodnet = GoodNet(path_bd, path_pruned)

    def load_image(self,infilename):
        img = Image.open(infilename)
        img.load()
        data = np.asarray(img, dtype="int32")
        return data

    def is_img_backdoored(self, img_path):
        x = self.load_image(img_path)
        x = x.astype(np.float32)
        x = np.array([x])
        label = self.goodnet.predict(x)
        return label

    def batch_eval(self,path_clean_x,path_valid_x_bd):
        bd_x_test, bd_y_test = data_loader(path_valid_x_bd)
        cl_x_test, cl_y_test = data_loader(path_clean_x)
        cl_label_p = self.goodnet.predict(cl_x_test)
        clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test)) * 100
        print('Clean Classification accuracy:', clean_accuracy)
        bd_label_p = self.goodnet.predict(bd_x_test)
        asr = np.mean(np.equal(bd_label_p, bd_y_test)) * 100
        print('Attack Success Rate:', asr)

if __name__ == '__main__':
    path_bd = 'E:\pycharmProjects\CSAW-HackML-2020\lab3\models\\bd_net.h5'
    path_pruned = 'E:\pycharmProjects\MachineLearningSecurity2021Lab3\models\pruned_bd_net_by_10_acc=84.43751623798389_asr=77.015675067117.h5'
    path_valid_x_bd = 'E:\pycharmProjects\CSAW-HackML-2020\lab3\data\\bd\\bd_valid.h5'
    path_clean_x = 'E:\pycharmProjects\CSAW-HackML-2020\lab3\data\cl\\valid.h5'
    eval = Eval( path_bd, path_pruned)
    label = eval.is_img_backdoored('E:\pycharmProjects\CSAW-HackML-2020\lab3\data\\test_img\\bd_0.jpeg')
    print(label)
