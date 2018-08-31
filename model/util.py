# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:14:52 2018

@author: 2014_Joon
"""

import numpy as np
#np.seterr(divide='ignore', invalid='ignore')  #ignore divide by zero or non
import matplotlib.pyplot as plt

from scipy import ndimage

from sklearn.metrics import confusion_matrix #classification_report
import itertools  # for confusion matrix plot

from PIL import Image as pil_image
import cv2

#import tensorflow as tf
#from tensorflow.python.framework import ops

from sklearn.model_selection import train_test_split
import dlib

import os
import glob
import numpy as np
import pandas as pd

from keras import backend as K
import keras

from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))       

K.set_learning_phase(False)

class_label = ['angry', 'happy', 'neutral']
n_class = len(class_label)
img_size = 48

data_path = '../data'
# to match the size, dimension, and color channel.

#landmarks = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(landmarks)

def dlib_face_coordinates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return detector(gray, 0)

def crop_face(frame, face_coordinates):
    # case : dlib
    cropped_img = frame
    (x, y, w, h) = face_coordinates
    cropped_img = frame[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
    # cv2.imwrite('./0.png', cropped_img, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

    # case : haar cascade
    # cropped_img = frame
    # for (x, y, w, h) in face_coordinates:
    #     cropped_img = frame[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
    return cropped_img

def preprocess(img, face_coordinates, face_shape=(48, 48)):
    face = crop_face(img, face_coordinates)
    face_resize = cv2.resize(face, face_shape)
    face_gray = cv2.cvtColor(face_resize, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./cropped.png', face_gray, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
    return face_gray

def preprocess_for_grad_CAM(img_arr, color_ch = 1):
    
    img_tensor = np.expand_dims(img_arr, 0)
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

def plot_grad_cam(model, img, pred_class, layer_idx = -3, n_layer =1, color_ch = 1):
    
    img_arr, img_tensor = preprocess_for_grad_CAM(img, color_ch)
            
    #pred_class = np.argmax(model.predict(img))
    n_row = int(1/3*n_layer)
    if n_layer%3 !=0:
        
        n_col = int(n_layer/n_row+1)
    else:
        n_col = int(n_layer/n_row)
        
    fig, axes = plt.subplots(n_row ,n_col, figsize=(10, 15))
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
        #top1_value = np.round(float(pred_values[top1]*100), decimals = 4)
        #axes[0].imshow(img_arr, cmap = 'gray')
        axes[0].imshow(img_arr, cmap = 'gray')
        axes[0].imshow(cam, cmap = 'jet', alpha = 0.5)
    
        #axes[0].axis('off')
        axes[0].axis('off')

    #axes[0].set_title("Pred:{}{}%\n True:{}\n".format(class_label[top1], top1_value, class_label[pred_class] ), fontsize=10)
                
    fig.tight_layout()
    fig.show()
    
    fig.savefig('gradCAM'+class_label[pred_class]+'.png',bbox_inches='tight',dpi=300)
    print(class_label[pred_class])
    

 # ex) img, cam, predictions = grad_cam(ak_net_0, img_path, class_idx, -13)
def load_sample_img(data_path):
    
    split_label = ['train','validation','test']
    class_label = ['angry', 'happy','neutral']
    
    list_dir = os.listdir(data_path)
    
    np.random.shuffle(list_dir)
    path_test = os.path.join(data_path, list_dir[0],split_label[2])
    print('\nRandom selection\n Subject:'+list_dir[0])
    
    samples =[]
    
    for i, i_dir in enumerate(class_label):
        path_class = os.path.join(path_test,i_dir)
        print('\nRandom selection from:'+ path_class)
        files = glob.glob(path_class+'/*.png')
        np.random.shuffle(files)
    
        img = preprocess_img(files[0])
        
        samples.append(img)
    return samples

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
    
    list_dir = os.listdir(data_path)
    for i, i_name in enumerate(list_dir):
        print(i_name+'\n')
        name_path = os.path.join(data_path,i_name)
        for j, j_split in enumerate(split_label):
            split_path = os.path.join(name_path, j_split)
            for k, k_class in enumerate(class_label):
                final_path = os.path.join(split_path, k_class)
                files = glob.glob(final_path+'/*.png')
                for file in files:
                    img = preprocess_img(file)
                    x.append(img)
                    y.append([k, j, i])  # class, split, subject
                    
    x=np.array(x)
    y=np.array(y)
    y=y[:,0]
    print(x.shape)
    print(y.shape)
    np.save('x_data_7.npy', x)
    np.save('y_data_7.npy', y)
                    
    return x, y
                

def plot_samples_from_path(data_path='/github/fer/data', class_idx=0):
    
    split_label = ['train','validation','test']
    class_label = ['angry', 'happy','neutral']
 
    path_test = os.path.join(data_path, split_label[2])
    #class_list_dir = os.listdir(path_test)
   
    path_class = os.path.join(path_test, class_label[0]) ####### 0: neutral 1: happy 2: angry
    files = glob.glob(path_class+'/*.png')
    #random.shuffle(files)
    #print(path_class)
    
    # temporary for test
    path_npy = '/python/autokeras/'
    os.chdir(path_npy)
    x = np.load('x_data_7.npy')
    y = np.load('y_data_7.npy')
    
    
    files = x[y==class_idx]
    np.random.shuffle(files)
    
    #files = files[0:100]
    n_fig = len(files)
    print(n_fig)
    
    n_row = int(np.sqrt(n_fig))
    n_col = int(n_fig/n_row)+1
    fig, axes = plt.subplots(n_row ,n_col)#, figsize=(10, 15))
    axes = axes.flatten()
    
    n_diff = n_row*n_col-n_fig
        
    for i, file in enumerate(files):
        #img = preprocess_img(file)
        
        axes[i].imshow(file, cmap = 'gray')
        axes[i].axis('off')
    for j in range(n_diff):
        axes[n_fig+j].axis('off')
    
        
    #axes[0].set_title("Class:{}".format(layer_idx_), fontsize=10)    
    fig.show()
    fig.savefig('total_'+class_label[class_idx]+'.png',bbox_inches='tight',dpi=300) 
    return 


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
    #os.chdir('./models/')
    
#    ## for whole data plot
#    data_path = '/github/fer/data'
#    for i in range(3):
#        plot_samples_from_path(class_idx=i)
   
    #x,y = load_img_save_npy(data_path)
            
    print("===============================================================")
    plt.close()
    loaded_model = load_model('./models/ak7_16.h5')    
    
#    #### transfer mobiel net model
#    model_path = 'test_mobile_model'
#    mobile_weight_path = 'test_mobile_weight'
#    
##    with CustomObjectScope({'relu6': keras.layers.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
##        loaded_model = load_model(model_path+'.h5')  
##    
##    loaded_model.load_weights(mobile_weight_path+'.h5')  
##    
    loaded_model.compile(loss = categorical_crossentropy,
              optimizer=Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
              metrics=['accuracy'])
    
    total_layer = len(loaded_model.layers)
    print('Total Layer:{}'.format(total_layer))
    
    #######
    samples = load_sample_img(data_path)
    
    n_layer = 15
    layer_idx = -4  # investigate layer start from ..
    color_ch = 1    # 1 for gray, 3 for model use RGB 
    print(len(samples))
    for i, sample in enumerate(samples):
        plot_grad_cam(loaded_model, sample, pred_class=i, layer_idx = layer_idx, n_layer=n_layer, color_ch = color_ch)
        


