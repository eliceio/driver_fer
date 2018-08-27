# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:15:05 2018

@author: 2014_Joon_IBS
"""

import os
import cv2
import pandas as pd
import numpy as np
import dlib

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

def normalize_resize(data, target_size = 197):
    #print(np.shape(data))
    new_shape = (target_size, target_size)
    
    for i, face in enumerate(data):
        #print(i, np.shape(face))

        if i==0:
            face = face.reshape(48,48)/255.0
            faces = cv2.resize(face, new_shape)
            faces = faces.reshape(1, target_size, target_size, 1)
            #print(np.shape(faces))
        else:
            face = face.reshape(48,48)/255.0
            face = cv2.resize(face, new_shape)
            face = face.reshape(1,target_size, target_size, 1)
            #print(np.shape(face))
            #print(np.shape(faces))
            faces = np.concatenate((faces, face), axis=0)
        
        if i%100==0:
            print(np.shape(faces))
            
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
    
    model.save_weights(path + model_name + '.h5')     
    # Save model as PNG (picture)
    plot_model(model, to_file = model_name + '_net.png', show_shapes=True, show_layer_names=True)
    
    print('Model & Weight are saved separatedly. (model: .json) (weight: .h5)')
    
### load autokeras result, get best model (torch), convert and return keras model.

def load_autokeras(path='python/autokeras/'):
    apple = ak.image_classifier.ImageClassifier(verbose = True, searcher_args={'trainer_args':{'max_iter_num':1}}, path = './', resume=True)
    searcher = apple.load_searcher()
    #searcher.history
    #apple.path
    best_id = apple.get_best_model_id
    print(best_id)
    graph = searcher.load_best_model()
    
    # Or you can get the best model by ID
    
    #graph = searcher.load_model_by_id(best_id)
    
    #torch_model = graph.produce_model()
    
    keras_model = graph.produce_keras_model()   # convert model from torch to keras
    save_model_weight(keras_model)

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

def convert_csv_to_dlib():
    print("convert csv to face dlib data start.\n")
    print('Dlib face generation start.\n')
    detector = dlib.get_frontal_face_detector() #Face detector
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
    
    fer = pd.read_csv(path + 'fer2013.csv')   
    pictures = np.array([np.fromstring(x, np.uint8, sep=' ') for x in fer.pixels.values])
    labels = np.array([np.fromstring(x, np.uint8) for x in fer.emotion.values]) #fromstring 대신 frombuffer 로 바꾸기. deprecate
    labels = labels[:,0]
 
   
    if not os.path.exists(path + 'dlib/'):
        os.makedirs(path + 'dlib/')
        
    for c_i, class_i in enumerate(class_label):
        if not os.path.exists(path + 'dlib/' + class_i): # if there's no class folder, make it
            os.makedirs(path + 'dlib/' + class_i)    
                
    landmarks = []
    for i, picture in enumerate(pictures): 
        xlist = []
        ylist = []
        picture = picture.reshape((48, 48))
        img_name = 'dlib_%d.jpg' % i
        path_name = os.path.join(path, 'dlib/', class_label[labels[i]], img_name)
        #path_name = path + class_label[labels[i]] + img_name
        #cv2.imwrite(path_name, picture)
        
 
        ## dlib save
        #frame = cv2.imread(path_name)
        
        gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        detections = detector(clahe_image, 1) #Detect the faces in the image
        for k,d in enumerate(detections): #For each detected face
            shape = predictor(clahe_image, d) #Get coordinates
            for i in range(1,68): # To draw 68 landmarks from dlib
                cv2.circle(picture, (shape.part(i).x, shape.part(i).y), 1, (0,0,200), thickness=1) 
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))
                #For each point, draw a red circle with thickness2 on the original frame
                
            
            for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
                landmarks.append(x)
                landmarks.append(y)
    
        #cv2.imwrite(os.path.join(path,class_label[labels[i]], img_name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(path_name, picture, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == "__main__":
    print('Something start')
    #cvt_csv2face()
    #load_autokeras()
#    x_train, x_test, y_train, y_test = load_data()
#    x_train = normalize_resize(x_train)
#    x_test = normalize_resize(x_test)
#    
#    np.save('./x_train_197.npy', x_train)
#    np.save('./x_test_197.npy', x_test)
#    np.save('./y_train.npy', y_train)
#    np.save('./y_test.npy', y_test)
    model = load_model('/python/ak_3class_transfer.h5')
    model_name = 'ak_3class_transfer'
    plot_model(model, to_file = model_name + '_net.png', show_shapes=True, show_layer_names=True)
    
    
    