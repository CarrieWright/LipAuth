# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 21:01:02 2018

@author: 691459
"""

import numpy as np
import scipy.io as sio

from LipAuth import LipAuth
from keras.callbacks import ModelCheckpoint, ProgbarLogger

from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq




def get_pairs(path, vidListFile):

    framesList = []
    labelsList = []
    for vf in open(vidListFile,'r').readlines():
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









def get_closest_pairs(embedded_vids, embedded_labels, vidNames, orig_vids):
    # embedded_vids is 128 embedding by num vids and labels (embedded_labels) to match
    # num_closest_negatives = number of negavtive examples per postive example

    final_pairs1 = []
    final_pairs2 = []
    final_labels = []
    final_vidNames = []
    final_01_labels = []
    
    embedded_vids = np.array(embedded_vids)
    embedded_vids = np.squeeze(embedded_vids)
    
    embedded_labels = np.array(embedded_labels)
    embedded_vidNames = np.array(vidNames)
    
    
    numVids = len(embedded_vids)
    
    # np.triu_indices(x,1) - every pair in x, choose 2
    ind1, ind2 = np.triu_indices(numVids, 1)
    
    stacked = np.stack([embedded_vids[ind1], embedded_vids[ind2]])
    stacked_labels = np.stack([embedded_labels[ind1], embedded_labels[ind2]])
    stacked_vidNames = np.stack([embedded_vidNames[ind1], embedded_vidNames[ind2]])
    
    #euclidean distance between every pair
    # distances is a vector containing the distances between every pair
    distances = np.linalg.norm(np.linalg.norm(stacked, axis=0), axis=1)
    
    
    # get all positive indicies
    positive_ind = np.where(np.array(stacked_labels[0]) == np.array(stacked_labels[1]) )
    print("number of possible pairs = ", len(distances))
    print("number of postive examples = ", len(positive_ind[0]))
    
    
    # get all -ve indicies to help find best -ve pair
    neg_ind = np.where(np.array(stacked_labels[0]) != np.array(stacked_labels[1]))
    print("number of possible negative examples = ", len(neg_ind[0]))
    
    
    # positive_ind+neg_ind (should = ) = total num possible pairs (len(ind1))
    for i in range(len(positive_ind[0])):
 #       print("length of list currently = ", len(final_pairs))
        
        ind_val = positive_ind[0][i]
#        print("positive index value : ", ind_val)
        
        # getting the actual video name
        currentVid1 = stacked_vidNames[0][ind_val]
        currentVid2 = stacked_vidNames[1][ind_val]
        
        # getting the index on the actual video in the original video list
        currentVid1indexNameList= vidNames.index(currentVid1.strip())
        currentVid2indexNameList= vidNames.index(currentVid2.strip())
        
        #add the actual video to the final_pairs list
        #final_pairs2.append([orig_vids[currentVid2indexNameList][np.newaxis]])
        #final_pairs1.append([orig_vids[currentVid1indexNameList][np.newaxis]]) 
        final_pairs1.append([orig_vids[currentVid1indexNameList]]) 
        final_pairs2.append([orig_vids[currentVid2indexNameList]])
        
        
        #final_pairs.append([stacked[0][ind_val], stacked[1][ind_val]])
        final_labels.append([stacked_labels[0][ind_val ], stacked_labels[1][ind_val]])
        final_vidNames.append([currentVid1, currentVid2])
        final_01_labels.append(1)
        
        closest_dist = np.inf
        single_pair1 = []
        single_pair2 = []
        single_labels = []
        single_vidNames = []
        
        
        pos_dist = distances[ind_val]
        
#        print("names of current positive video pair: ", currentVid1, \
#              " and ", currentVid2, " with a distance of : ", pos_dist)

        
        #loop through all negative pairs
        for j in range(len(neg_ind[0])):
            neg_ind_val = neg_ind[0][j]
            
            # get the unique video name of the 2 videos in the pair
            neg_names = [stacked_vidNames[0][neg_ind_val], stacked_vidNames[1][neg_ind_val]]
            # if the name matches with first video in original pair, get the 
            # distance so can check it!
            if currentVid1 in neg_names:
                cur_dist = distances[neg_ind_val]
                #print("current dist", cur_dist)
                #print("closest dist", closest_dist)
                #print("positive dist",pos_dist)
                # the distance must be smaller than the current smallest AND 
                # AND larger than the positive example's distance
                if (cur_dist<closest_dist and cur_dist > pos_dist):
                    
#                    print("distance fitted the match")
#                    print("looking for negative for video : ", currentVid1, " and found it",\
#                      "in the pair : ", neg_names, " with a distance of : ", \
#                      cur_dist, " comparing it to positive distance : ", pos_dist, \
#                      " and closest existing distance of : ", closest_dist)

                    # it passed so reset all these values
                    closest_dist = cur_dist
                    single_labels = [stacked_labels[0][neg_ind_val],stacked_labels[1][neg_ind_val]]
                    #single_pair = [stacked[0][neg_ind_val],stacked[1][neg_ind_val]]  
                    single_vidNames = [stacked_vidNames[0][neg_ind_val],stacked_vidNames[1][neg_ind_val]]
                    
                    

                    indiciesOfVid1 = vidNames.index(single_vidNames[0].strip())
                    indiciesOfVid2 = vidNames.index(single_vidNames[1].strip())
                    
                    #single_pair1 = [orig_vids[indiciesOfVid1][np.newaxis], orig_vids[indiciesOfVid2][np.newaxis]]
                    single_pair1 = orig_vids[indiciesOfVid1]
                    single_pair2 = orig_vids[indiciesOfVid2]
                    
                    
                    
                    
                    
                    
         # after looping through all negativ pairs add the best neg pairs to our dataset          
        if (single_pair1 != []) :
            final_pairs1.append([single_pair1])
            final_pairs2.append([single_pair2])
            final_labels.append(single_labels)  
            final_vidNames.append(single_vidNames)
            final_01_labels.append(0)
        else :
            print("WARNING - NO NEGATIVE EXAMPLE FOR VIDEO : ", currentVid1)
        
    
    final_pairs = [final_pairs1, final_pairs2]
    print("final pairs length : ", len(final_pairs))    
    print("final labels length : ", len(final_labels)) 
    print("final vidNames length : ", len(final_vidNames))     
    return final_pairs, final_labels, final_vidNames, final_01_labels
        

def readInList(listAllVids, path):
    
    framesList = []
    labelsList = []
    namesList = []
    
    for vf in open(listAllVids,'r').readlines():
        namesList.append(vf.strip())
        data = sio.loadmat(path+vf.strip())
        frames = data['rgbFrames']
        frames = np.swapaxes(frames, 0, 1)
        frames = np.rollaxis(frames, 3, 0)
        framesList.append(frames)
        labelsList.append(vf.split('_')[0])

    return framesList, namesList, labelsList     


def writeOutPairs(epoch, listPairNames, destination):
    fileName = destination + "pairs_at_epoch_"+ str(epoch) + ".txt"
    
    f = open(fileName, 'w')
    for i in range(len(listPairNames)):
        f.write(listPairNames[i][0] + "  " + listPairNames[i][1] + "\n")

    f.close()    
    
    

def train(listAllTrainingVids, pathToVids, destination, weight_path, epochs=1000):
    
    lip_auth = LipAuth(weight_path=weight_path)
    
    # raw vid data not in pairs
    tr_framesList, tr_namesList, tr_labelsList = readInList(listAllTrainingVids, pathToVids)    

    
    for i in range(epochs):
        print("\nEpoch %d/%d" %(i+1,epochs))
        #get training data and labels
        
        embedded_tr_vids = []
        for vid in tr_framesList:
            
            embedded_tr_vid = lip_auth.lipAuth_embedding.predict(vid[np.newaxis])
            
            embedded_tr_vids.append(embedded_tr_vid)
            
        
        tr_pairs, tr_labels, tr_vidNames, tr_01_labels = get_closest_pairs(
                embedded_tr_vids, tr_labelsList, tr_namesList, tr_framesList)
        
        writeOutPairs(i, tr_namesList, destination)
        # get number training examples
        n_examples = len(tr_labels)
        bar = tqdm(range(n_examples))
        loss = 0
     
        print("n_examples: ", n_examples)
        print("tr_pairs length: ", len(tr_pairs))
        print("tr_pairs[0] length: ", len(tr_pairs[0]))
        
        for j in bar: # assumes batch size of 1
            bar.set_description('%d/%d'%(j,n_examples))
            
            
            loss += lip_auth.lipAuth.train_on_batch([np.array(tr_pairs[0][j]), \
                np.array(tr_pairs[1][j])], np.array([tr_01_labels[j]]))
            bar.set_postfix(loss=loss/float(j+1))              
                
        e = i+1
        if e <= 10:            
            print "at epoch: ", str(e), " saving model "
            name = destination + "weights_for_lipAuthModel_Epoch_" + str(e) +".h5"   
            lip_auth.lipAuth.save_weights(name)
        
        elif e%2==0:
            print "at epoch: ", str(e), " saving model "
            name = destination + "weights_for_lipAuthModel_Epoch_" + str(e) +".h5"   
            lip_auth.lipAuth.save_weights(name)
            


    return True





def train_more(listAllTrainingVids, pathToVids, destination, weight_path, lastcheckpoint, lastepoch, epochs=1000):
    
    lip_auth = LipAuth(weight_path=weight_path)
    lip_auth.lipAuth.load_weights(lastcheckpoint)
    
    # raw vid data not in pairs
    tr_framesList, tr_namesList, tr_labelsList = readInList(listAllTrainingVids, pathToVids)    

    
    for i in range(lastepoch, epochs):
        print("\nEpoch %d/%d" %(i+1,epochs))
        #get training data and labels
        
        embedded_tr_vids = []
        for vid in tr_framesList:
            
            embedded_tr_vid = lip_auth.lipAuth_embedding.predict(vid[np.newaxis])
            
            embedded_tr_vids.append(embedded_tr_vid)
            
        
        tr_pairs, tr_labels, tr_vidNames, tr_01_labels = get_closest_pairs(
                embedded_tr_vids, tr_labelsList, tr_namesList, tr_framesList)
        
        writeOutPairs(i, tr_namesList, destination)
        # get number training examples
        n_examples = len(tr_labels)
        bar = tqdm(range(n_examples))
        loss = 0
     
        print("n_examples: ", n_examples)
        print("tr_pairs length: ", len(tr_pairs))
        print("tr_pairs[0] length: ", len(tr_pairs[0]))
        
        for j in bar: # assumes batch size of 1
            bar.set_description('%d/%d'%(j,n_examples))
            
            
            loss += lip_auth.lipAuth.train_on_batch([np.array(tr_pairs[0][j]), \
                np.array(tr_pairs[1][j])], np.array([tr_01_labels[j]]))
            bar.set_postfix(loss=loss/float(j+1))              
                
        e = i+1
        if e <= 10:            
            print "at epoch: ", str(e), " saving model "
            name = destination + "weights_for_lipAuthModel_Epoch_" + str(e) +".h5"   
            lip_auth.lipAuth.save_weights(name)
        
        elif e%2==0:
            print "at epoch: ", str(e), " saving model "
            name = destination + "weights_for_lipAuthModel_Epoch_" + str(e) +".h5"   
            lip_auth.lipAuth.save_weights(name)
            


    return True

    
        
    
    

def main():
    
    weight_path = '/mnt/lvm/bigscratch/users/691459/Carrie/LipAuth/lipnet/unseen-weights178.h5'
    checkpoint_file = '/mnt/lvm/bigscratch/users/691459/Carrie/LipAuth/snapshots/weights_for_lipAuthModel_Epoch_38.h5'  
    vid_path = '/mnt/lvm/bigscratch/users/691459/Carrie/files50_100/'
    destination = "/mnt/lvm/bigscratch/users/691459/Carrie/LipAuth/snapshots/"
    epochs = 1000
    
    listAllVids = "/mnt/lvm/bigscratch/users/691459/Carrie/LipAuth/776training_194x4mats.txt"
    #listAllVids = "/mnt/lvm/bigscratch/users/691459/Carrie/LipAuth/10ppl_40vid_xm2_s1s2_mats.txt"
    
    lastepoch = 38 
    train_more(listAllVids, vid_path, destination, weight_path, checkpoint_file, lastepoch, epochs=epochs) 
    
    print("Finished Training")
    
    

if __name__ == '__main__':
    main()