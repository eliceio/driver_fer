# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 18:46:07 2018

@author: 2014_Joon_IBS
"""

import numpy as np
#np.seterr(divide='ignore', invalid='ignore')  #ignore divide by zero or non
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from skimage.measure import label, regionprops
#import scipy
from scipy import ndimage

from sklearn.metrics import confusion_matrix #classification_report
import itertools  # for confusion matrix plot

from PIL import Image as pil_image
import cv2

#import tensorflow as tf
#from tensorflow.python.framework import ops
from keras import backend as K


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
# for jupyter notebook enviornment. 


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix #classification_report
import itertools  # for confusion matrix plot

from PIL import Image as pil_image

K.set_learning_phase(False)

class_label = ['angry', 'happy', 'neutral']
n_class = len(class_label)
img_size = 48
# to match the size, dimension, and color channel.
    
def preprocess(img_path, color_ch = 1):
    
    img = pil_image.open(img_path)
    img = img.resize((img_size, img_size))
    img_arr = np.asarray(img) / 255.
    img_tensor = np.expand_dims(img_arr, 0)
    #img_tensor = np.expand_dims(img_tensor, 3)    
    img_tensor = np.stack((img_tensor,)*color_ch, -1 )  # to make fake RGB channel, color_ch =3, or 1 for gray
    print(img_tensor.shape)
    print(img_arr.shape)
    
    return img_arr, img_tensor

def grad_cam(model, img_arr, img_tensor, class_idx, layer_idx):

    

    y_c = model.layers[-1].output.op.inputs[0][0, class_idx]  # final layer (before softmax)
    layer_output = model.layers[layer_idx].output  # pick specific layer output (caution: conv layer only)
    
    grad = K.gradients(y_c, layer_output)[0]  # calculate gradient of y_c w.r.t. A_k from the conv layer output
    gradient_fn = K.function([model.input], [layer_output, grad, model.layers[-1].output])
    
    conv_output, grad_val, predictions = gradient_fn([img_tensor])
    conv_output, grad_val = conv_output[0], grad_val[0]
    
    weights = np.mean(grad_val, axis=(0, 1))
    cam = np.dot(conv_output, weights)
    cam = cv2.resize(cam, (img_size, img_size))
    
    # relu 
    cam = np.maximum(cam, 0)    
    cam = cam / (cam.max() + 1e-10) # To prevent divide by zero
    
    return cam, predictions

def plot_grad_cam(model, img, pred_class=2, layer_idx = -3, n_layer =1, color_ch = 1):
    
    img_arr, img_tensor = preprocess(img, color_ch)
        
    #img = np.expand_dims(img, 0)
    #pred_class = np.argmax(model.predict(img))
   
    #plt.close()
    n_row = 3
    fig, axes = plt.subplots(n_row ,int(n_layer/n_row))#, figsize=(10, 15))
    axes = axes.flatten()
    #for i in range(n_layer):
    
#top3 = np.argpartition(pred_values, -3)[-3:]  #top 4
    if n_layer >1:
        for i in range(n_layer):
            cam, predictions = grad_cam(model, img_arr, img_tensor,  pred_class, layer_idx-i) #in case of model class, model.model
            
            pred_values = np.squeeze(predictions, 0)
            top1 = np.argmax(pred_values)
            top1_value = np.round(float(pred_values[top1]*100), decimals = 4)
            layer_idx_i = layer_idx-i
            layer_name = model.layers[layer_idx_i].name
            #axes[0,i].imshow(img_arr, cmap = 'gray')
            axes[i].imshow(img_arr, cmap = 'gray')
            axes[i].imshow(cam, cmap = 'jet', alpha = 0.5)
            #axes[0,i].axis('off')
            axes[i].axis('off')
            axes[i].set_title("Layer idx:{}\n{}\n".format(layer_idx_i, layer_name), fontsize=10)        
    else:
        cam, predictions = grad_cam(model, img_arr, img_tensor,  pred_class, layer_idx) #in case of model class, model.model
        
        pred_values = np.squeeze(predictions, 0)
        top1 = np.argmax(pred_values)
        top1_value = np.round(float(pred_values[top1]*100), decimals = 4)
        #axes[0].imshow(img_arr, cmap = 'gray')
        axes[0].imshow(img_arr, cmap = 'gray')
        axes[0].imshow(cam, cmap = 'jet', alpha = 0.5)
    
        #axes[0].axis('off')
        axes[0].axis('off')

    #axes[0].set_title("Pred:{}{}%\n True:{}\n".format(class_label[top1], top1_value, class_label[pred_class] ), fontsize=10)
                
    fig.tight_layout()
    fig.show()
    
    fig.savefig('gradCAM'+class_label[pred_class]+'.png',bbox_inches='tight',dpi=100)
    

 # ex) img, cam, predictions = grad_cam(ak_net_0, img_path, class_idx, -13)

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
    
# ex) confusion_result = make_confusion_matrix(loaded_model, x_test, y_test, False)



if __name__ == "__main__":
    #os.chdir('/github')
    os.chdir('/python/models')
    print("===============================================================")
    loaded_model = load_model('ak_3class_transfer.h5')    
    #### transfer mobiel net model
#    model_path = 'transfer_model_20'
#    mobile_weight_path = 'cam_model_weight'
#    
#    with CustomObjectScope({'relu6': keras.layers.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
#        loaded_model = load_model(model_path+'.h5')  
#    
#    loaded_model.load_weights(mobile_weight_path+'.h5')  
#    
    #######
    
    os.chdir('/python/data')    
    
    img_path = 'angry.png'
    img_path1 = 'happy.png'
    img_path2 = 'neutral.png'
    
    loaded_model.compile(loss = categorical_crossentropy,
              optimizer=Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
              metrics=['accuracy'])
    
    total_layer = len(loaded_model.layers)
    print('Total Layer:{}'.format(total_layer))
    
    n_layer = 15
    layer_idx = -4  # investigate layer start from ..
    color_ch = 1    # 1 for gray, 3 for model use RGB 
    plot_grad_cam(loaded_model, img_path, pred_class=0, layer_idx = layer_idx, n_layer=n_layer, color_ch = color_ch)
    plot_grad_cam(loaded_model, img_path1, pred_class=1, layer_idx = layer_idx, n_layer=n_layer, color_ch = color_ch)
    plot_grad_cam(loaded_model, img_path2, pred_class=2, layer_idx = layer_idx, n_layer=n_layer, color_ch = color_ch)

