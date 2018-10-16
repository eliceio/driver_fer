# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 02:04:40 2018

@author: 2014_Joon_IBS
"""

import autokeras as ak
from autokeras.preprocessor import OneHotEncoder, DataTransformer
from autokeras.constant import Constant

from keras.models import model_from_json
from keras.models import load_model
from keras.utils import np_utils

import cv2
import os
import numpy as np
from torchvision import models
import pandas as pd
#from autokeras.classifier import load_image_dataset
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
import glob

#class_label = ['angry', 'disgust', 'fear','happy','sad','surprise','neutral']
class_label = ['angry', 'happy','neutral']
target_size = 48    # img_size
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


def preprocess_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  #load img as grayscale
    img = clahe.apply(img)  # histogram equalization
    img = np.array(img)/255.  # normalize
    return img

def load_img_save_npy(data_path):
    
    split_label = ['test','train','validation']
    
    class_label = ['angry', 'happy','neutral']
    
    x = []
    y = []
    
    #os.walk
    #list_dir = os.listdir(data_path)
#    for i, i_name in enumerate(list_dir):
#        print(i_name+'\n')
#        name_path = os.path.join(data_path,i_name)
    for j, j_split in enumerate(split_label):
        split_path = os.path.join(data_path, j_split)
        for k, k_class in enumerate(class_label):
            final_path = os.path.join(split_path, k_class)
            files = glob.glob(final_path+'/*.png')
            for file in files:
                img = preprocess_img(file)
                x.append(img)
                y.append(k)  # class, split, subject
                    
    x=np.array(x)
    y=np.array(y)
    
    print(x.shape)
    print(y.shape)
    np.save('x_data.npy', x)
    np.save('y_data.npy', y)
                    
    return x, y

def model_fit(x_train, y_train, resume = True, iter_num=10, time_limit = 12):
    print('Model training start')
    ##clf = ak.ImageClassifier(verbose = True, searcher_args={'trainer_args':{'max_iter_num':10}}, path = './', resume=resume) 
    clf = ak.image_classifier.ImageClassifier(verbose = True, searcher_args={'trainer_args':{'max_iter_num':iter_num}}, path = './', resume=resume) 

    print('fit start')
    out = clf.fit(x_train, y_train, time_limit = time_limit*60*60 )
    
    print('out: ', out)
    return clf
    
def final_fit(clf, x_train, y_train, x_test, y_test, iter_num =5):
    print('final fit')
    final_out = clf.final_fit(x_train, y_train, x_test, y_test, retrain=False, trainer_args={'max_iter_num':iter_num})
    print('final out: ', final_out)
 
    results = clf.predict(x_test)
    print('predict: ', results)

    y = clf.evaluate(x_test, y_test)
    print('eval: ', y)

def show_and_save_best_model(clf, model_name = 'ak_2_1'):
    best_model_id = clf.get_best_model_id()
    print('\nBest model id: {}\n '.format(best_model_id))
    best_model = clf.load_searcher().load_best_model()
    print('n_layer: ', best_model.n_layers)
    print(best_model.produce_model())
    keras_model = best_model.produce_keras_model()
    #save_model_weight(keras_model)
    keras_model.save('./' + model_name + '.h5')
    print('\nSave best model.\n')
    
    plot_model(keras_model, to_file = model_name + '_net.png', show_shapes=True, show_layer_names=True)
    
    return keras_model

def load_ak_show_best_id():
    clf = ak.image_classifier.ImageClassifier(verbose = True, 
                                          searcher_args={'trainer_args':{'max_iter_num':1}}, path = './', resume=True)     
    
    best_model_id = clf.get_best_model_id()
    print('Best model id: {} '.format(best_model_id))    

if __name__ == '__main__':
    
    #print(os.getcwd())
    #x_train, x_test, y_train, y_test = load_data()
    
    data_path = '../data/heonyeong/'
    #os.chdir(data_path) # change directory. autokeras wiil be stored here.
    # 1. data load
    print('\n#####################\nload data\n')
    #load_img_save_npy(data_path)
    print('\n#####################\nSave as npy\n')
     
    class_label = ['background', 'body','nose', 'tail']
    
    #data_path = '/Data/_backup/img_fer_ck_cam_3class/'
    
    #### aug
    #data_path = '/github/fer/data/jungjoon/test/angry/'
    #data_path = '/Data/mice_processed'
    
          
    data_path = '/Data/mice_processed/'
    os.chdir(data_path)
    list_files = os.listdir(data_path)
    
    x_data = np.load('./x_data_mice.npy')  # 7 means total 7 members, not 7 emotion classes :)
    y_data = np.load('./y_data_mice.npy')    
    print('\n#####################\nload npy\n')
    os.chdir('/python/autokeras/')
    
    target_size=40
    
    x_data = x_data.reshape(-1, target_size, target_size,1)
    
    class_dist = [len(y_data[y_data==i]) for i, c in enumerate(class_label)] 
    
    print('\nClass distribution:{0}\n'.format(class_dist))
    # train +val / test set split 
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, shuffle = True, random_state=33)
    
    # Autokeras fit start.
    clf = model_fit(x_train, y_train, resume=True, iter_num=10, time_limit = 12)
    
    print('\nFinal fit start.\n')
    final_fit(clf, x_train, y_train, x_test, y_test)
    show_and_save_best_model(clf, model_name='ak_model_mice.h5')