#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 22:29:39 2018

@author: Carrie
"""
import scipy.io as sio
import numpy as np

from lipnet.model2 import LipNet
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, Lambda, Dense
from keras import backend as K


import keras

keras.backend.set_image_data_format('channels_last')
print(keras.backend.image_data_format())

class LipAuth(object):
    
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=75, weight_path=""):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)
        self.weight_path = weight_path
        self.build()

    """        
    def build(self):
        lipnet = LipNet(img_c= self.img_c, img_w= self.img_w, img_h= self.img_h,\
                        frames_n= self.frames_n, absolute_max_string_len=32, \
                        output_size=28)
        
        lipnet.model.load_weights(self.weight_path)

        
        self.base_network = Model(lipnet.model.get_layer('the_input').input, lipnet.model.get_layer('bidirectional_2').output)

        input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)
        print(input_shape)
        
        input_a = Input(shape=input_shape, dtype='float32')
        input_b = Input(shape=input_shape, dtype='float32')
        
        embedded_a = self.base_network(input_a)
        embedded_b = self.base_network(input_b)
        
        embedded_a = Bidirectional(GRU(64, return_sequences=False, \
          kernel_initializer='Orthogonal', name='gru3'), merge_mode='concat')(embedded_a)
        embedded_b = Bidirectional(GRU(64, return_sequences=False, \
          kernel_initializer='Orthogonal', name='gru3'), merge_mode='concat')(embedded_b)
        
        distance = Lambda(euclidean_distance,
                     output_shape=eucl_dist_output_shape)([embedded_a, embedded_b])
        
        sigmoid = Dense(1, activation='sigmoid')(distance)
        
        self.model = Model([input_a, input_b], sigmoid)
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
    """


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
            print(counter)
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
    
 
    
if __name__ == '__main__':
     print('mani') 
     epchs = 20
     training_data = sio.loadmat('10ppl_40vid_s1s2_Pairs780.mat')
     training_pairs = training_data['pairs'] 
     training_pairedlabels = np.squeeze(training_data['pairedlabels'])
     training_order = np.random.permutation(training_pairs.shape[0])
     
     training_pairs = training_pairs[training_order]
     training_pairedlabels = training_pairedlabels[training_order]
     
     print(training_pairs.shape)
     training_pairs = np.swapaxes(training_pairs, 2, 5)
     training_pairs = np.swapaxes(training_pairs, 4, 5)
     training_pairs = np.swapaxes(training_pairs, 0, 1)
     print(training_pairs.shape)
     
     
     cv_data = sio.loadmat('10ppl_40vid_s3_Pairs_190.mat')
     cv_pairs = cv_data['pairs']
     cv_pairedlabels = np.squeeze(cv_data['pairedlabels'])
     cv_order = np.random.permutation(cv_pairs.shape[0])
     
     cv_pairs = cv_pairs[cv_order]
     cv_pairedlabels = cv_pairedlabels[cv_order]
     print(cv_pairs.shape)
     cv_pairs = np.swapaxes(cv_pairs, 2, 5)
     cv_pairs = np.swapaxes(cv_pairs, 4, 5)
     cv_pairs = np.swapaxes(cv_pairs, 0, 1)
     print(cv_pairs.shape)
     
     
     
     lip_auth = LipAuth(weight_path='/Users/Carrie/git/LipNet/evaluation/models/unseen-weights178.h5')
     lip_auth.lipAuth.fit([training_pairs[0], training_pairs[1]], training_pairedlabels, 
                          validation_data=([cv_pairs[0], cv_pairs[1]], 
                                           cv_pairedlabels), epochs=epchs)
     lip_auth.lipAuth.summary()
     
     lip_auth.lipAuth.save("20epoch_model_save_10peopleAllPairs_4vidsTrain780pairs_2cr190pairs.h5")
     lip_auth.lipAuth.save_weights('20epoch_model_save_weights_10peopleAllPairs_4vidsTrain_2cr.h5')

     
     
     
     
     
     
 
"""
if __name__ == '__main__':
    print('main')
    data = sio.loadmat('10peoplePairsMat_780.mat')
    pairs = data['pairs'] 
    pairedlabels = np.squeeze(data['pairedlabels'])
    order = np.random.permutation(pairs.shape[0])
    pairs = pairs[order]
    pairedlabels = pairedlabels[order]
    print(pairs.shape)
    # originally (pairs, no pairs, h, w, c, frames ) ?? think
    # new shape (no pairs, no vids 2, noFrames, width?, height?, channels)
    pairs = np.swapaxes(pairs, 2, 5)
    pairs = np.swapaxes(pairs, 4, 5)
    pairs = np.swapaxes(pairs, 0, 1)
    print(pairs.shape)
    
    lip_auth = LipAuth(weight_path='/Users/Carrie/git/LipNet/evaluation/models/unseen-weights178.h5')
    
    lip_auth.lipAuth.fit([pairs[0][:700], pairs[1][:700]], pairedlabels[:700],
      validation_data=([pairs[0][700:], pairs[1][700:]], pairedlabels[700:]), epochs=5)
"""
