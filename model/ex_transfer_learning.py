# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 21:23:09 2018

@author: 2014_Joon_IBS
"""

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19 

from keras.layers import Dense
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

from skimage.transform import resize 
import numpy as np
import pandas as pd
import cv2


###### parameter
class_label = ['angry', 'disgust', 'fear','happy','sad','surprise','neutral']
n_class = len(class_label)

path = './'  # location
file_name = 'fer2013.csv'

img_size = 48  # fer data size. 48 x 48
target_size = 48 #197 # minimum data size for specific net such as Inception, VGG, ResNet ...
#n_data = 1000  # to deal with memory issue..

epochs = 20  # n of times of training using entire data

#####

def load_data():
    print("Start load_data")
    fer = pd.read_csv(path + file_name)
    fer_train = fer[fer.Usage == 'Training']
    fer_test = fer[fer.Usage.str.contains('Test', na=False)]

    x_train = np.array([list(map(int, x.split())) for x in fer_train['pixels'].values])
    y_train = np.array(fer_train['emotion'].values)

    x_test = np.array([list(map(int, x.split())) for x in fer_test['pixels'].values])
    y_test = np.array(fer_test['emotion'].values)
    
    # to deal with the memory issue.. 
    #x_train = x_train[0:n_data]
    #x_test = x_test[0:n_data]
    #y_train = y_train[0:n_data]
    #y_test = y_test[0:n_data]
    
    x_train = normalize_x(x_train)
    x_test = normalize_x(x_test)
    
    y_train = np_utils.to_categorical(y_train, n_class)
    y_test = np_utils.to_categorical(y_test, n_class)

    return x_train, x_test, y_train, y_test

def normalize_x(data):
    faces = []
    
    for face in data:
        face = face.reshape(img_size, img_size) / 255.0
        face = cv2.resize(face, (target_size, target_size))
        
        faces.append(face)
        #np.concatenate(faces,face)
    faces = np.asarray(faces)
    #del(data)
    
    #faces = np.expand_dims(faces, -1)
    faces = np.stack((faces,)*3, -1 )  # to make RGB channel
    return faces

## load data

if __name__ == "__main__":
    print("===============================================================")

    x_train, x_test, y_train, y_test = load_data()
    #np.shape(x_train)
    
    ######### load model for transfer learning. ex) ResNet50
    # include_top must be False, to be replaced by our own classifier / softmax
    # you can use other pooling method such as max, or whatever, and even add more layers
    # input shape cannot be reduced what we want...
    print("===============================================================")
#    print('Transfer learning.\n\n Model: ResNet50')
#    model_resnet = ResNet50(include_top = False, weights = 'imagenet', input_tensor = None, 
#                     input_shape = (target_size, target_size, 3), pooling = 'avg' , classes = n_class)
    
    print('Transfer learning.\n\n Model: VGG19\n')
    model_vgg = VGG19(include_top = False, weights = 'imagenet', input_tensor = None, 
                     input_shape = (target_size, target_size, 3), pooling = 'avg' , classes = n_class)

    ##### Add new output for fine-tuning
    x = model_vgg.output
    #x = Dense(2048, activation ='relu')(x)
    predictions = Dense(n_class, activation ='softmax')(x)
    model = Model(inputs = model_vgg.input, outputs = predictions)

    ##### Freeze all layers in the resnet. we just use the pretrained weight, during training, no back propagation will appear here.
    # But you can selectively activate specific layer for training.
    for layer in model_vgg.layers:
        layer.trainable = False


    ##### Start learning. same procedure.
    model.compile(loss = categorical_crossentropy,
                      optimizer=Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
                      metrics=['accuracy'])

    hist = model.fit(x_train, y_train, 
                              validation_split = 0.2, 
                              shuffle = True, 
                              batch_size = 16, epochs = epochs, verbose = 0, 
                               )

    print('\nEnd of training')