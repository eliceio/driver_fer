# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 21:23:09 2018

@author: 2014_Joon_IBS
"""


from keras.preprocessing import image

from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19 
from keras.applications.resnet50 import ResNet50

from keras.layers import Dense
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline  
# for jupyter notebook enviornment. 

import itertools  # for confusion matrix plot

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix #classification_report

###### parameter
#class_label = ['angry', 'disgust', 'fear','happy','sad','surprise','neutral']
class_label = ['angry', 'happy','neutral']

n_class = len(class_label)

path = './'  # location
os.chdir(path)

img_size = 48  # fer data size. 48 x 48
target_size = 48 #197 # minimum data size for specific net such as Inception, VGG, ResNet ...

epochs = 2  # n of times of training using entire data
batch_size = 16

### select model to use. you have to import first. ex) from keras.applications.vgg19 import VGG19 
model_transfer = VGG19(include_top = False, weights = 'imagenet', input_tensor = None, 
                     input_shape = (target_size, target_size, 3), pooling = 'avg' , classes = n_class)

# class weight. give more weight to angry 
class_weight = {0: 2.,
                1: 1.,
                2: 1.}

###################

##### arrange data class and reshape the data
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
    print('After normalize x:{a} y:{b}\n'.format(a= xx.shape, b=yy.shape))     

    return xx, yy

def sample_plot(x_test, y_test):
    x_angry = x_test[y_test==0]
    a_f = np.squeeze(x_angry[1])
    plt.imshow(a_f, cmap='gray')
    
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
    plt.savefig('loss_accuracy_plot')
    plt.close()

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
    plt.close()

if __name__ == "__main__":
    print("===============================================================")
    
#    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state=33)
#    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.25, shuffle = True, random_state=33)
#    print('N of train:{} val:{} test:{}\n'.format(len(x_train),len(x_val),len(x_test)))
#        
    # data load
    x_data = np.load('./x_data.npy')
    y_data = np.load('./y_data.npy')
        
    # 2. arrange the data. shape change, use specific class only, ...
    x_data, y_data = data_arrange(x_data, y_data)

    # 3. train / test split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, shuffle = True, random_state=33)
    #np.shape(x_train)
    
    # 4. data batch generation. Parameter belows include augmentations, so please check before run
    datagen = ImageDataGenerator(
        
        zca_whitening=True  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.2,  # set range for random shear
        zoom_range=0.2,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=1./255,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0)

    datagen.fit(x_train)
    
        

    # early stop condition to prevent overfitting
    
    early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0.001, patience = 20, 
                                       verbose = 1, mode = 'max')
    
    
    ######### load model for transfer learning. ex) ResNet50
    # include_top must be False, to be replaced by our own classifier / softmax
    # you can use other pooling method such as max, or whatever, and even add more layers
    # input shape cannot be reduced what we want...
    print("===============================================================")
#    print('Transfer learning.\n\n Model: ResNet50')
#    model_resnet = ResNet50(include_top = False, weights = 'imagenet', input_tensor = None, 
#                     input_shape = (target_size, target_size, 3), pooling = 'avg' , classes = n_class)
    
    print('Transfer learning.\n\n Model: VGG19\n')

    ##### Add new output for fine-tuning
    x = model_transfer.output
    #x = Dense(2048, activation ='relu')(x)
    predictions = Dense(n_class, activation ='softmax')(x)
    model = Model(inputs = model_transfer.input, outputs = predictions)

    ##### Freeze all layers in the resnet. we just use the pretrained weight, during training, no back propagation will appear here.
    # But you can selectively activate specific layer for training.
    for layer in model_transfer.layers:
        layer.trainable = False

    ##### Start learning. same procedure.
    
    model.compile(loss = categorical_crossentropy,
                      optimizer=Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
                      metrics=['accuracy'])

    hist = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch = len(x_train)/batch_size, epochs=epochs,
                        workers=1, use_multiprocessing=False, class_weight=class_weight, 
                        validation_data=(x_test, y_test),  callbacks = [early_stopping] )
    
    ## 2nd learning. now, train all layers
    print('\nEnd of 1st training')
    
    for layer in model_transfer.layers:
        layer.trainable = True

    ##### Start learning. same procedure.
    
    model.compile(loss = categorical_crossentropy,
                      optimizer=Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
                      metrics=['accuracy'])

    hist = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch = len(x_train)/batch_size, epochs=epochs,
                        workers=1, use_multiprocessing=False, class_weight=class_weight, 
                        validation_data=(x_test, y_test),  callbacks = [early_stopping] )
    

    print('\nEnd of 2nd training')
    
    plot_hist(hist)            
    confusion_result = make_confusion_matrix(model, x_test, y_test, normalize = True) 
        
    model.save('./transfer_model.h5')