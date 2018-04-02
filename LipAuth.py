#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 22:29:39 2018

@author: Carrie
"""
import scipy.io as sio
import numpy as np

from lipnet.model2 import LipNet
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, Lambda, Dense
from keras import backend as K

#import keras

#keras.backend.set_image_data_format('channels_last')
#print(keras.backend.image_data_format())


class LipAuth(object):
    
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=None, weight_path=""):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)
        self.weight_path = weight_path
        self.build()


    def build(self):
        
        self.buildEmbeddingModel()
                
        input_a = Input(shape=self.input_shape, dtype='float32')
        input_b = Input(shape=self.input_shape, dtype='float32')
        
        embedded_a = self.lipAuth_embedding(input_a)
        embedded_b = self.lipAuth_embedding(input_b)
                
        distance = Lambda(self.euclidean_distance,
                          output_shape=self.eucl_dist_output_shape)([embedded_a, embedded_b])
                          
        sigmoid = Dense(1, activation='sigmoid')(distance)
        
        self.lipAuth = Model([input_a, input_b], sigmoid)
        self.lipAuth.compile(loss='binary_crossentropy', optimizer='adam')
     
    def buildEmbeddingModel(self):
        lipnet = LipNet(img_c= self.img_c, img_w= self.img_w, img_h= self.img_h,\
                        frames_n= self.frames_n, absolute_max_string_len=32, \
                        output_size=28)
        
        lipnet.model.load_weights(self.weight_path)
        
        lipnet.model.summary()
              
        model = Model(lipnet.model.get_layer('the_input').input, \
                      lipnet.model.get_layer('bidirectional_2').output)
        
        # Freeze all layers and compile the model
        counter = 0
        for layer in model.layers:
            layer.trainable = False
            counter +=1
            #print(counter)
            if counter > 20:
                layer.backward_layer.trainable = False  
                layer.forward_layer.trainable = False        
    
        x = model.output
        x = Bidirectional(GRU(64, return_sequences=False, \
                    kernel_initializer='Orthogonal', name='gru3'), merge_mode='concat')(x)
        
        self.lipAuth_embedding = Model(model.input, x)
        self.lipAuth_embedding.summary()

    def euclidean_distance(self, vects):
        x, y = vects
        dist = K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
        return dist

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)
    
