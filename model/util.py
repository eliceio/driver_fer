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

K.set_learning_phase(False)

class_label = ['angry', 'disgust', 'fear', 'happy', 'sad','surprise', 'neutral']
img_size = 48
# to match the size, dimension, and color channel.
    
def preprocess(img_path):
    
    img = pil_image.open(img_path)
    img = img.resize((img_size, img_size))
    img_arr = np.asarray(img) / 255.
    img_tensor = np.expand_dims(img_arr, 0)
    img_tensor = np.expand_dims(img_tensor, 3)    
    
    return img_arr, img_tensor


def grad_cam(model, img_path, class_idx, layer_idx):

    img_arr, img_tensor = preprocess(img_path)

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
    cam = cam / cam.max()
    
    return img_arr, cam, predictions

def plot_grad_cam(model, img, pred_class=1, layer_idx = -5):
        
    img = np.expand_dims(img, 0)
    #pred_class = np.argmax(model.predict(img))
   
    plt.figure(2)
    fig, axes = plt.subplots(1, 2, figsize=(30, 24))
    img, cam, predictions = grad_cam(model, img, pred_class, layer_idx) #in case of model class, model.model
        
    pred_values = np.squeeze(predictions, 0)
    top1 = np.argmax(pred_values)
    top1_value = np.round(float(pred_values[top1]*100), decimals = 4)
    top4 = np.argpartition(pred_values, -4)[-4:]  #top 4
    
    axes[0, 0].set_title("Pred:{}{}%\n True:{}\n{}".format(class_label[top1], top1_value, class_label[pred_class], top4 ), fontsize=10)
    axes[0, 0].imshow(img,cmap = 'gray')
    axes[0, 1].imshow(img,cmap = 'gray')
    axes[0, 1].imshow(cam, cmap = 'jet', alpha = 0.5)

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
