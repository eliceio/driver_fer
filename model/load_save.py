# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:15:05 2018

@author: 2014_Joon_IBS
"""

import os
import cv2
import pandas as pd
import numpy as np

from keras.models import model_from_json
from keras.models import load_model
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

import autokeras as ak

class_label = ['angry', 'disgust', 'fear','happy','sad','surprise','neutral']
n_class = len(class_label)

path = '/python/DDSA/fer/fer2013/'
file_name = 'fer2013_2.csv'

def load_data():
    print("Start load_data")
    fer = pd.read_csv(path + file_name)
    fer_train = fer[fer.Usage == 'Training']
    fer_test = fer[fer.Usage.str.contains('Test', na=False)]

    x_train = np.array([list(map(int, x.split())) for x in fer_train['pixels'].values])
    y_train = np.array(fer_train['emotion'].values)

    x_test = np.array([list(map(int, x.split())) for x in fer_test['pixels'].values])
    y_test = np.array(fer_test['emotion'].values)

    x_train = normalize_x(x_train)
    x_test = normalize_x(x_test)
    y_train = np_utils.to_categorical(y_train, n_class)
    y_test = np_utils.to_categorical(y_test, n_class)

    return x_train, x_test, y_train, y_test

# load data from csv for autokeras.
# no need to use utils.to_categorical
def load_data_for_ak():
    print("Start load_data")
    fer = pd.read_csv(path + file_name)
    fer_train = fer[fer.Usage == 'Training']
    fer_test = fer[fer.Usage.str.contains('Test', na=False)]

    x_train = np.array([list(map(int, x.split())) for x in fer_train['pixels'].values])
    y_train = np.array(fer_train['emotion'].values)

    x_test = np.array([list(map(int, x.split())) for x in fer_test['pixels'].values])
    y_test = np.array(fer_test['emotion'].values)

    x_train = normalize_x(x_train)
    x_test = normalize_x(x_test)

    return x_train, x_test, y_train, y_test

def normalize_x(data):
    faces = []

    for face in data:
        face = face.reshape(48, 48) / 255.0
        face = cv2.resize(face, (48, 48))
        faces.append(face)

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    return faces


#from autokeras.preprocessor import OneHotEncoder, DataTransformer
#from autokeras.constant import Constant
### To get autokeras,
# pip install git+https://github.com/jhfjhfj1/autokeras.git

### load pretrained modle
# 1. load model & weight together (.h5)
# or, 2. load model (.json), weight (.h5) separately

## ex) loaded_model = load_model_weight('sm_net.json', 'sm_net.h5')
def load_model_weight(model_name, weight_name = None):

    if weight_name ==None:  # which means, model & weight are saved in one file        
        loaded_model = load_model(model_name)
        print('Model & weight are loaded altogether.')
    else:
        with open(model_name, 'r') as m:
            loaded_model_json = m.read()
        loaded_model = model_from_json(loaded_model_json)   # load model architecture from json
        loaded_model.load_weights(weight_name)        
        print('Model & weight are loaded separately.')
    
    return loaded_model            

### Save model / weight separately
## save_model_and_weight(loaded_model, 'new_test23', './nnn/f2/')
def save_model_weight(model, model_name = 'kafka', path = './'):
    if not os.path.exists(path):
        os.makedirs(path)    
        
    model_json = model.to_json() # save model architecture as json 

    with open(path + model_name+'.json','w')as m:
        m.write(model_json)
    
    model.save_weights(path + model_name + 'h5')     
    # Save model as PNG (picture)
    plot_model(model, to_file = model_name + '_net.png', show_shapes=True, show_layer_names=True)
    
    print('Model & Weight are saved separatedly. (model: .json) (weight: .h5)')
    
### load autokeras result, get best model (torch), convert and return keras model.

def load_autokeras(path='/tmp/face/'):
    apple = ak.image_classifier.ImageClassifier(verbose = True, searcher_args={'trainer_args':{'max_iter_num':1}}, path = path, resume=True)
    searcher = apple.load_searcher()
    #searcher.history
    #apple.path
    
    graph = searcher.load_best_model()
    
    # Or you can get the best model by ID
    #best_id = apple.get_bset_model_id
    #graph = searcher.load_model_by_id(best_id)
    
    #torch_model = graph.produce_model()
    
    keras_model = graph.produce_keras_model()   # convert model from torch to keras

    return keras_model


## convert csv file to face picture (png / jpg)
def cvt_csv2face():

    fer = pd.read_csv(path + file_name)
    pictures = np.array([np.fromstring(x, np.uint8, sep=' ') for x in fer.pixels.values])
    labels = np.array([np.fromstring(x, np.uint8) for x in fer.emotion.values]) #fromstring will be deprecated. change to frombuffer.
    labels = labels[:,0]    # class label data    
    
    for _, j_class in enumerate(class_label):
        if not os.path.exists(path + j_class):
            os.makedirs(path + j_class)
        
    for i, picture in enumerate(pictures):
        picture = picture.reshape((48, 48))
        img_name = '%d.png' % i
        #plt.show(picture)
        cv2.imwrite(os.path.join(path, class_label[labels[i]], img_name), picture, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #params for PNG, low value means low compression, big file size.(0 to 9)
