# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:15:05 2018

@author: 2014_Joon_IBS
"""

import os

from keras.models import model_from_json
from keras.models import load_model

import autokeras as ak
#from autokeras.preprocessor import OneHotEncoder, DataTransformer
#from autokeras.constant import Constant
### To get autokeras,
# pip install git+https://github.com/jhfjhfj1/autokeras.git

### load pretrained modle
# 1. load model & weight together (.h5)
# or, 2. load model (.json), weight (.h5) separately

## ex) loaded_model = load_model_weight('sm_net.json', 'sm_net.h5')
def load_model_weight(model_name, weight_name = None):
    print('ddd')
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