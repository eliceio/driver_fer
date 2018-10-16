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
import glob

from keras.models import model_from_json
from keras.models import load_model
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split

import autokeras as ak

import shutil

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))      
class_label = ['angry', 'happy','neutral']
#class_label = ['angry', 'disgust', 'fear','happy','sad','surprise','neutral']
#split_label = ['train', 'validation', 'test']
split_label = ['train', 'test']
n_class = len(class_label)

path = '/data/'
file_name = 'fer2013_processed.csv'


def put_dir(now_path):  # to put all different folders (ex- test, train, val, ...) into just distinct classes
    count = 0 
    count2 = 0
    for (path, dir, files) in os.walk(now_path):
        print('\n{}'.format(count))
        count += 1
        
        emotion_bool = [i in path for i in class_label] ## temp[i] 에 i 가 속해있기만 하면 true 
        #emotion_bool.index(true)
        
        if any(emotion_bool): # 어떤 emotion 폴더 감지되면
            idx_emotion = emotion_bool.index(True)
            new_path = os.path.join(now_path,class_label[idx_emotion])
            if not os.path.exists(new_path):  # 폴더 없으면 생성
                os.makedirs(new_path)
            print(len(files))
            for file in files:
                new_file = os.path.join(new_path,file)
                src_file = os.path.join(path,file)
                if os.path.isfile(new_file):
                    count2 +=1
                    print('\n')
                    print(count2)
                    new_file = new_path + '/new_'+ str(count2)+file
                    print(new_file)
                shutil.copy2(src_file, new_file)
            
        
    
    

def balance_dist(now_path = './', ratio = 0.35):
    #files = glob.glob(final_path+'/*.png')
    
    
    for (path, dir, files) in os.walk(now_path):
        n_files = len(files)
        n_final = int(n_files*ratio)   # file 을 이만큼 남기고 지울것
        
        #n_final = 4000
        np.random.shuffle(files)
        for filename in files:
            
            ext = os.path.splitext(filename)[-1]
            if ext == '.png':
                #print("%s/%s" % (path, filename))
                fullname = os.path.join(path, filename)
                os.remove(fullname)
                
            n_current = len(glob.glob(path+'/*.png'))
            if n_current <= n_final:  # 원한만큼 지우면 그만.
                print('\n dist:{}\n'.format(n_current))
                break
    
    

def make_img_split(path):
    
    os.chdir(path)
    
    for k, k_class in enumerate(class_label):
        print('Class:{}\n'.format(k_class))
        final_path = os.path.join(data_path, k_class)
        files = glob.glob(final_path+'/*.png')
        
        n_files = len(files)
       # idx = np.arange(n_files)
        np.random.shuffle(files)
        
        f_train = files[0 : int(n_files*0.6)]
        f_val = files[int(n_files*0.6):int(n_files*0.8)]
        f_test = files[int(n_files*0.8):n_files]       
        
        
        for tf in f_train:
            t_path = os.path.join(path, split_label[0], k_class)
            if not os.path.exists(t_path):
                os.makedirs(t_path)
            shutil.copy(tf, t_path)
        for vf in f_val:
            v_path = os.path.join(path, split_label[1], k_class)
            if not os.path.exists(v_path):
                os.makedirs(v_path)
            shutil.copy(vf, v_path)
        for ttf in f_test:
            tt_path = os.path.join(path, split_label[2], k_class)
            if not os.path.exists(tt_path):
                os.makedirs(tt_path)
            shutil.copy(ttf, tt_path)
        

def preprocess_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  #load img as grayscale
    img = clahe.apply(img)  # histogram equalization
    img = np.array(img)/255.  # normalize
    
    return img

def load_faces_save_npy(data_path):
    
    #class_label = ['angry', 'disgust', 'fear', 'happy', 'neutral','sad', 'surprise']
    
    x = []
    y = []
    
    #os.walk
    
    for k, k_class in enumerate(class_label):
        final_path = os.path.join(data_path, k_class)
        files = glob.glob(final_path+'/*.*')
        for file in files:
            img = preprocess_img(file)
            x.append(img)
            y.append(k)  # class, subject
            
    x=np.array(x)
    y=np.array(y)
    
    print(x.shape)
    print(y.shape)
    np.save('x_data_fer_ck_cam.npy', x)
    np.save('y_data_fer_ck_cam.npy', y)
    print(os.getcwd())
                    
    return x, y

def load_fer_ck_save_npy(data_path):
    
    #class_label = ['angry', 'disgust', 'fear', 'happy', 'neutral','sad', 'surprise']
    
    x = []
    y = []
    
    #os.walk
    list_dir = os.listdir(data_path)
    for i, i_name in enumerate(list_dir):
        print(i_name+'\n')
        name_path = os.path.join(data_path,i_name)     
        for k, k_class in enumerate(class_label):
            final_path = os.path.join(name_path, k_class)
            files = glob.glob(final_path+'/*.png')
            for file in files:
                img = preprocess_img(file)
                x.append(img)
                y.append([k, i])  # class, subject
                
    x=np.array(x)
    y=np.array(y)
    y=y[:,0]
    
    for i in range(6):
         print(len(y[y==i]))
    print(x.shape)
    print(y.shape)
    
    np.save('x_data_ferckcam.npy', x)
    np.save('y_data_ferckcam.npy', y)
                    
    return x, y


# load csv fer2013 data for FER.
def load_csv_data():
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

# selecte specific class, normalize, class distribution balance, make fake RGB channel, y to one hot encoding

def data_arrange(x_data,y_data, color_ch = 3, img_size = 48):
    # data class re-arrange
    
    # Angry vs neutral case
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
    xx = np.stack((xx,)*color_ch, -1 )  # 1 for gray, 3 for making fake RGB channel
    yy = np_utils.to_categorical(yy, n_class)
    print('After normalize x:{a} y:{b}\n'.format(a= xx.shape, b=yy.shape))     

    return xx, yy

# just simple reshape and normalize
def normalize_x(data):
    faces = []

    for face in data:
        face = face.reshape(48, 48) / 255.0
        face = cv2.resize(face, (48, 48))
        faces.append(face)

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    return faces

# normalize and resize. Needs a lot of memory, space and time. But sometims (ex- ResNet transfer) we need it.
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

def gray_to_3ch(input_data):
    temp = input_data[:,:,:]
    for i in range(1,3):
        input_data = np.concatenate((input_data,temp), axis=3)
        print(i)
    print(np.shape(input_data))
    return input_data

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

# make face picture with dlib 68 face landmarks from csv fer2013 data. on-going.
        


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
    #data_path = '/Data/fer_ck_cam_3_img/_fer_ck/'
    data_path = '/Data/_backup/img_fer_ck_cam_3class/'
    #data_path = '/github/fer/data/ta_total/'
    os.chdir(data_path)
    #make_img_split(data_path)
    
    #load_fer_ck_save_npy(data_path)
    
    load_faces_save_npy(data_path)
    
    
    #d_p = '/Data/fer_ck_cam_3_img/webcam_data/'
    #d_p = '/Data/fer_ck_cam_3_img/fer_3class/'
    #balance_dist(now_path = d_p)
    #put_dir(d_p)    
#    list_dir = os.listdir(d_p)
#    for d in list_dir:
#        put_dir(os.path.join(d_p,d))

#    model = load_model('/python/ak_3class_transfer.h5')
#    model_name = 'ak_3class_transfer'
#    plot_model(model, to_file = model_name + '_net.png', show_shapes=True, show_layer_names=True)
#    
    
    