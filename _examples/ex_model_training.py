# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 13:49:18 2018

@author: 2014_Joon_IBS
"""
import os
import glob
import numpy as np
import pandas as pd
import cv2

from sklearn.metrics import confusion_matrix #classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.models import load_model

from keras.callbacks import EarlyStopping, TensorBoard

import matplotlib.pyplot as plt
#%matplotlib inline  # for jupyter notebook enviornment. 

import itertools  # for confusion matrix plot

from PIL import Image as pil_image


#import tensorflow as tf
#from tensorflow.python.framework import ops
from keras import backend as K

#path = '/python/DDSA/fer/fer2013/'
path = '/data/'  # path of the csv file
file_name = 'fer2013_processed.csv'
class_label = ['angry', 'disgust', 'fear','happy','sad','surprise','neutral']
n_class =len(class_label)
img_size = 48  #pixel size of the data input
os.chdir(path)
epoch = 2 # epoch means entire data set. if we set 100 as an epoch, entire data set will be trained (by batch) 100 times.
# due to the slow speed, default setting is 2. you can increase if you have GPU.
    

# load data from csv

def load_data():
    fer = pd.read_csv(path + file_name)
    fer_train = fer[fer.Usage == 'Training']
    fer_test = fer[fer.Usage.str.contains('Test')]

    x_train = np.array([list(map(int, x.split())) for x in fer_train['pixels'].values])
    y_train = np.array(fer_train['emotion'].values)

    x_test = np.array([list(map(int, x.split())) for x in fer_test['pixels'].values])
    y_test = np.array(fer_test['emotion'].values)

    # normalize for x
    x_train = normalize_x(x_train)
    x_test = normalize_x(x_test)
    
    y_train = np_utils.to_categorical(y_train, n_class)  # to convert one-hot encoding
    y_test = np_utils.to_categorical(y_test, n_class)

    return x_train, x_test, y_train, y_test

# simple normalize for load data
def normalize_x(data):
    faces = []

    for face in data:
        face = face.reshape(img_size, img_size) / 255.0
        face = cv2.resize(face, (img_size, img_size))
        faces.append(face)

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    return faces 
    
### Save model / weight separately
## save_model_and_weight(loaded_model, 'new_test23', './nnn/f2/')
def save_model_weight(model, model_name = 'kafka', path = './'):  # this is for 'class' model. 
    if not os.path.exists(path):
        os.makedirs(path)    
        
    model_json = model.to_json() # save model architecture as json 

    with open(path + model_name+'.json','w')as m:
        m.write(model_json)
    
    model.save_weights(path + model_name + '.h5')     
    # Save model as PNG (picture)
    plot_model(model, to_file = model_name + '_net.png', show_shapes=True, show_layer_names=True)
    
    print('Model & Weight are saved separatedly. (model: .json) (weight: .h5)')

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

    # Plot loss / accuracy of training / validation data history.
    # To check whether the learning is ok or not.    
def plot_hist(hist):
    plt.figure(0)
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='lower left')

    plt.show()
    fig.savefig('loss_accuracy_plot')
    

# Make and plot confusion matrix. To see the detailed imformation about TP, TN of each classes.    
def make_confusion_matrix(model, x, y, normalize = True):
    predicted = model.predict(x)

    pred_list = []; actual_list = []
    for i in predicted:
        pred_list.append(np.argmax(i))
    for i in y:
        actual_list.append(np.argmax(i))

    confusion_result = confusion_matrix(actual_list, pred_list)
    plot_confusion_matrix(confusion_result, classes = class_label, normalize = normalize, title = 'Confusion_matrix')
    return confusion_result

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):  
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("normalized")
    else:
        print('without normalization')

    print(cm)
    plt.figure(1)
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_matrix')
 

#    
class BaseNet:
    def __init__(self, weights_file=None):
        self.model = self.buildNet(weights_file)

    def buildNet(self, weights_file=None):
        print("Start buildNet.")
        model = Sequential()
        model.add(Conv2D(64, (5,5), padding='valid', input_shape=(48,48,1))) 
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), padding='valid'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Conv2D(128, (4, 4), padding='valid'))
        model.add(Flatten())
        model.add(Dense(3072))
        model.add(Dense(7))
        model.add(Activation('softmax'))    

        self.model = model

        print("Create model successfully!")
        if weights_file:
            model.load_weights(weights_file)

        model.compile(loss = categorical_crossentropy,
                      optimizer=Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
                      metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs = 10):
        print("Start training")
        # early stopping to prevent overfitting.
        early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0.001, patience = 30, 
                                       verbose = 1, mode='max')
    
        # fit (learning) start. 1-epoch means entire data set.
        # during training, monitor the validation accuracy using validation data (splitted from training data ex -  0.2).
        hist = self.model.fit(x_train, y_train, 
                              validation_split = 0.2, 
                              shuffle = True, 
                              batch_size = 32, epochs = epochs, verbose = 1, 
                              callbacks = [early_stopping] )
        return hist
    
    # evaluate the trained model / weight using test data (not training data)
    def evaluate(self, x_test, y_test):
        print("evaluate test.")
        scores = self.model.evaluate(x_test, y_test, batch_size = 32)
        print("Loss:", scores[0])
        print("Accuracy:", scores[1])
        return scores

    def predict(self, x):
        return self.model.predict(x)

    def plot(self):
        plot_model(self.model, to_file = 'BaseNet.png', show_shapes = True, show_layer_names = True)
        self.model.summary()  

if __name__ == "__main__":
    print("===============================================================")
    
    print("loading datasets")
    x_train, x_test, y_train, y_test = load_data()
    
    ###
    
    # Net
    if os.path.isfile('apple.h5'):  # if there's already pretrained model / weights, load them
        print('\n Saved model & weight found. \n 2nd (or + a) training start. \n')
        loaded_model = load_model_weight(model_name = 'apple.json', weight_name = 'apple.h5')
        loaded_model.compile(loss = categorical_crossentropy,
                      optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
                      metrics = ['accuracy'])
        early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0.001, patience = 30, 
                                       verbose = 1, mode = 'max')
    
        # fit (learning) start. 1-epoch means entire data set.
        # during training, monitor the validation accuracy using validation data (splitted from training data ex -  0.2).
        hist = loaded_model.fit(x_train, y_train, 
                              validation_split = 0.2, 
                              shuffle = True, 
                              batch_size = 32, epochs = epoch, verbose = 1, 
                              callbacks = [early_stopping] )
        scores = loaded_model.evaluate(x_test, y_test, batch_size = 32)    
        model = loaded_model        
        
    else:  # if there's no existing model / weight, build new model
        print("1st baseNet\n")                 
        basenet = BaseNet()    # new model.
        # training.         
        hist = basenet.train(x_train, y_train, epoch)  # train during n epoch. hist is to plot loss, accuracy plot        
        scores = basenet.evaluate(x_test, y_test)  #scores for calculate test loss and accruacy        
        model = basenet.model
        
    print("Loss:", scores[0])
    print("Accuracy:", scores[1])
    
    plot_hist(hist)            
    confusion_result = make_confusion_matrix(model, x_test, y_test, normalize = True) 
            
    model_name = 'apple.h5'
    #save_model_weight(model, model_name = 'apple', path = './')
    model.save(model, model_name)
        # This is for saving the model as png file. 
    #test_fig = plot_model(model, to_file='BaseNet.png', show_shapes=True, show_layer_names=True)
    #plt.plot(test_fig)    
 

    
    