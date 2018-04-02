import numpy as np
import scipy.io as sio

from LipAuth import LipAuth
from keras.callbacks import ModelCheckpoint, ProgbarLogger

import progressbar



def get_pairs(path, vidListFile):

    framesList = []
    labelsList = []
    for vf in open(path+vidListFile,'r').readlines():
        data = sio.loadmat(path+vf.strip())
        frames = data['rgbFrames']
        frames = np.swapaxes(frames, 0, 1)
        frames = np.rollaxis(frames, 3, 0)
        framesList.append(frames)
        labelsList.append(vf.split('_')[0])

    pairs1 = []
    pairs2 = []
    labels = []
    numExamples = len(labelsList)
    for i in range(numExamples):
        for j in range(numExamples):
            if not np.array_equal(framesList[i], framesList[j]):
                if j>i:
                    pairs1.append([framesList[i]])
                    pairs2.append([framesList[j]])
                    if labelsList[i] == labelsList[j]:
                        labels += [1]
                    else:
                        labels += [0]

    pairs = [pairs1, pairs2]
    return pairs, labels

def train(pairs, labels, validation_data, checkpointFile, epochs=1000):
    
    lip_auth = LipAuth(weight_path='/Users/Carrie/git/LipNet/evaluation/models/unseen-weights178.h5')
  
    #callbacks = [ModelCheckpoint(checkpointFile, save_best_only=True, save_weights_only=True)]

    
    # setting loss to really big number so the start will be lower
    best_loss = np.inf
    for i in range(epochs):
        barLength = len(labels)
        print("\nEpoch %d/%d" %(i,epochs))
        bar = progressbar.ProgressBar().start()
        
        
        
        print("\nloss on training: ", best_loss)
        
        for j in range(barLength): # assumes batch size of 1
            
            #print('%d/%d'%(j+1, len(labels)))
            best_loss += lip_auth.lipAuth.train_on_batch([np.array(pairs[0][j]), \
            np.array(pairs[1][j])], np.array([labels[j]]))
            #print(i,loss)
            bar.update(j/780)
        
        bar.finish()
            
            
            

def main():
    path = '/Users/Carrie/Desktop/xm2demo/mats_75/'
  
  # get training data
    pairs, labels = get_pairs(path, 'training.txt')
  
    # get validation data
    validation_data = get_pairs(path, 'validation.txt')
  
    checkpointFile = 'lipAuth_test_best_checkpoint.h5'

    train(pairs, labels, validation_data, checkpointFile, epochs=10)
    
    

if __name__ == '__main__':
    main()
