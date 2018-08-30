# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:48:41 2018

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

import keras
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

from keras.callbacks import EarlyStopping, TensorBoard

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import itertools  # for confusion matrix plot


class_label = ['angry', 'happy','neutral']

n_class = len(class_label)

img_size = 48  # fer data size. 48 x 48
target_size = 48 #197 # minimum data size for specific net such as Inception, VGG, ResNet ...

epochs = 10  # n of times of training using entire data
batch_size = 16

########################### You have to designate the location

your_name = 'jungjoon' # your data folder
fer_ck_path = '/data/'  # location of x_data, y_data

#############
data_path = '../data/'+your_name+'/'
model_path = '../model/models/test_mobile_model.h5'  # location

#os.chdir('/github/drive_fer/webcam/')


def plot_hist(hist):
    
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

    fig.show()
    plt.savefig('loss_accuracy_plot')


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
    fig = plt.figure(1)
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
    fig.savefig('Confusion_matrix')
    fig.show()
    
def data_arrange(x_data,y_data):
    # data class re-arrange
    
    # Angry vs neutral vs happy case
    x_angry = x_data[y_data==0]
    x_happy = x_data[y_data==3]
    x_neutral = x_data[y_data==6]
    
    y_angry = y_data[y_data==0]
    y_happy = y_data[y_data==3]
    y_neutral = y_data[y_data==6]
    
    # number of happy samples are twice  
    # To avoid class distribution bias. use only 50% sample of happy class
    x_happy_use, x_no, y_happy_use, y_no = train_test_split(x_happy, y_happy, test_size = 0.5, shuffle = True, random_state=33)
    
    #print('Before normalize:{a}\n'.format(a= x_angry[0]))
    xx = np.concatenate((x_angry, x_happy_use, x_neutral),axis=0)/255.0 #concatenate & normalized
    yy = np.concatenate((y_angry, y_happy_use, y_neutral), axis=0)
    yy[yy==3]=1
    yy[yy==6]=2
    
    xx = xx.reshape(-1, img_size,img_size)
    xx = np.stack((xx,)*3, -1 )  # to make fake RGB channel
    yy = np_utils.to_categorical(yy, n_class)
    print('After preprocess \n x:{a} y:{b}\n'.format(a= xx.shape, b=yy.shape))     

    return xx, yy

def sample_plot(x_test, y_test):
    x_angry = x_test[y_test==0]
    a_f = np.squeeze(x_angry[1])
    plt.imshow(a_f, cmap='gray')
    
    
if __name__ =="__main__":
    print('\n############### Load FER, CK data for test... ########\n')
    # data load to test
    x_data = np.load(fer_ck_path + 'x_data.npy')
    y_data = np.load(fer_ck_path +'y_data.npy')
    
    # 2. arrange the data. shape change, use specific class only, ...
    x_data, y_data = data_arrange(x_data, y_data)
    
    # 3. train / test split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, shuffle = True, random_state=33)
    #np.shape(x_train)
    
    ##### Load Mobile net trained with fer + ck data 
    print('\n############### Load Mobile net trained with fer, ck data... ########\n')
    
    with CustomObjectScope({'relu6': keras.layers.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
        loaded_model = load_model(model_path)    
        
    # To confirm that loaded model has about 0.8 accuracy, plot confusion matrix
    loaded_model.compile(loss = categorical_crossentropy,
                      optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
                      metrics = ['accuracy'])

    print('\n############### Confusion matrix result preparing... ########\n')
    
    confusion_result = make_confusion_matrix(loaded_model, x_test, y_test, normalize = True) 
    
    #####
    early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0.001, patience = 20, 
                                       verbose = 1, mode = 'max')
    
    print('\n############### Preparing webcam data to augment...  ########\n')
    
    # for on-line augmentation
    train_datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0)
      
    test_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow_from_directory(
            data_path+'train',
            target_size=(48, 48),
            batch_size=16, #color_mode = 'grayscale',
            class_mode='categorical', save_to_dir = data_path+'aug', save_prefix='aug')  #save_to_dir option is for monitoring data augmentation
    
    validation_generator = test_datagen.flow_from_directory(
            data_path+'validation',
            target_size=(48, 48),
            batch_size=16,
            class_mode='categorical')
    
    early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0.001, patience = 20, 
                                           verbose = 1, mode = 'max')
    
    # make augmentation folders. augmented filed will be saved here.
    if not os.path.exists(data_path + '/aug'): # if there's no class folder, make it
        os.makedirs(data_path + 'aug')  
        
    print('\n############### Training start. ########\n')
    
    ##### Training
    hist = loaded_model.fit_generator(
        train_generator,
        steps_per_epoch=30,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=10,callbacks = [early_stopping])
    
    ##### plot the history and save the model weight
    print('\n############### Result plot. save the model weight. ########\n')
    
    plot_hist(hist)
    os.chdir('../model/models/')
    loaded_model.save_weights('test_mobile_weight.h5')
