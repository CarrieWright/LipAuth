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


def get_closest_pairs(embedded_vids, embedded_labels, embedded_vidNames):
    # embedded_vids is 128 embedding by num vids and labels (embedded_labels) to match
    # num_closest_negatives = number of negavtive examples per postive example

    final_pairs = []
    final_labels = []
    final_vidNames = []
    final_01_labels = []
    
    embedded_vids = np.array(embedded_vids)
    embedded_labels = np.array(embedded_labels)
    embedded_vidNames = np.array(embedded_vidNames)
    
    numVids = len(embedded_vids)
    
    # np.triu_indices(x,1) - every pair in x, choose 2
    ind1, ind2 = np.triu_indices(numVids, 1)
    
    stacked = np.stack([embedded_vids[ind1], embedded_vids[ind2]])
    stacked_labels = np.stack([embedded_labels[ind1], embedded_labels[ind2]])
    stacked_vidNames = np.stack([embedded_vidNames[ind1], embedded_vidNames[ind2]])
    
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
        
        currentVid1 = stacked_vidNames[0][ind_val]
        currentVid2 = stacked_vidNames[1][ind_val]
        
        final_pairs.append([stacked[0][ind_val], stacked[1][ind_val]])
        final_labels.append([stacked_labels[0][ind_val ], stacked_labels[1][ind_val]])
        final_vidNames.append([currentVid1, currentVid2])
        final_01_labels.append(1)
        
        closest_dist = np.inf
        single_pair = []
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
                    single_pair = [stacked[0][neg_ind_val],stacked[1][neg_ind_val]]  
                    single_vidNames = [stacked_vidNames[0][neg_ind_val],stacked_vidNames[1][neg_ind_val]]
                    
         # after looping through all negativ pairs add the best neg pairs to our dataset          
        if (single_pair != []) :
            final_pairs.append(single_pair)
            final_labels.append(single_labels)  
            final_vidNames.append(single_vidNames)
            final_01_labels.append(0)
        else :
            print("WARNING - NO NEGATIVE EXAMPLE FOR VIDEO : ", currentVid1)
        
        
    print("final pairs length : ", len(final_pairs))    
    print("final labels length : ", len(final_labels)) 
    print("final vidNames length : ", len(final_vidNames))     
    return final_pairs, final_labels, final_vidNames, final_01_labels
        

def readInList(listAllVids, path):
    
    framesList = []
    labelsList = []
    namesList = []
    
    for vf in open(path+listAllVids,'r').readlines():
        namesList.append(vf.strip())
        data = sio.loadmat(path+vf.strip())
        frames = data['rgbFrames']
        frames = np.swapaxes(frames, 0, 1)
        frames = np.rollaxis(frames, 3, 0)
        framesList.append(frames)
        labelsList.append(vf.split('_')[0])

    return framesList, namesList, labelsList     


def writeOutPairs(epoch, listPairNames, destination):
    fileName = destination + "pairs_at_epoch_"+ epoch + ".txt"
    
    f = open(fileName, 'w')
    for i in range(len(listPairNames)):
        f.write(listPairNames[i][0] + "  " + listPairNames[i][1] + "\n")

    f.close()    
    
    

def train(listAllVids, listValidationData, pathToVids, destination, epochs=1000):
    
    lip_auth = LipAuth(weight_path='/Users/Carrie/git/LipNet/evaluation/models/unseen-weights178.h5')
    
    # raw vid data not in pairs
    tr_framesList, tr_namesList, tr_labelsList = readInList(listAllVids, pathToVids)
    # validation data
    val_pairs, val_labels = get_pairs(pathToVids, listValidationData)
    num_val_examples = len(val_labels)
    

    # setting loss to really big number so the start will be lower
    best_loss = np.inf
    EER_at_low_val = np.inf
    epoch_at_low_val = 0
    threshold_at_low_val = 0
    
    for i in range(epochs):
        print("\nEpoch %d/%d" %(i+1,epochs))
        #get training data and labels
        embedded_tr_vids = lip_auth.lipAuth_embedding.predict(tr_framesList)
        tr_pairs, tr_labels, tr_vidNames, tr_01_labels = get_closest_pairs(
                embedded_tr_vids, tr_labelsList, tr_namesList)
        
        writeOutPairs(i, tr_namesList, destination)
        # get number training examples
        n_examples = len(tr_labels)
        bar = tqdm(range(n_examples))
        loss = 0
        val_loss = 0
     
        for j in bar: # assumes batch size of 1
            bar.set_description('%d/%d'%(j,n_examples))
            loss += lip_auth.lipAuth.train_on_batch([np.array(tr_pairs[0][j]), \
                np.array(tr_pairs[1][j])], np.array([tr_01_labels[j]]))
            bar.set_postfix(loss=loss/float(j+1))
        
        scores = []
        for k in range(num_val_examples):
            val_loss += lip_auth.lipAuth.test_on_batch([np.array(val_pairs[0][k]), \
                                                        np.array(val_pairs[1][k])], \
                                                       np.array([val_labels[k]]))
            pred = lip_auth.lipAuth.predict_on_batch([np.array(val_pairs[0][k]), \
                                                        np.array(val_pairs[1][k])])
            scores.append(pred[0][0])
    

        FPR, TPR, thresholds = roc_curve(np.array(val_labels), np.array(scores))   
        EER = brentq(lambda x : 1. - x - interp1d(FPR, TPR)(x), 0., 1.)
        threshold = interp1d(FPR, thresholds)(EER)

                
                
        print('validation loss : %.3lf' % (val_loss/float(num_val_examples)))
        print("EER = ", EER)
        print("Threshold = ", threshold)
        
        
        if val_loss/float(num_val_examples) < best_loss:
            best_loss = val_loss/float(num_val_examples)
            print("validation loss decreased ... EER changed from : ", EER, " to : ", EER_at_low_val)
            
            EER_at_low_val = EER
            epoch_at_low_val = i+1
            threshold_at_low_val = threshold

            
            name = destination + "weights_for_lipAuthModel_EER_" + str(EER_at_low_val) + \
            "_threshold_" + str(threshold_at_low_val) + "_atEpch" + str(epoch_at_low_val) +".h5"
            lip_auth.lipAuth.save_weights(name)



    return EER_at_low_val, threshold_at_low_val, epoch_at_low_val

    
        
    
    
    #
    #
    #
    
    
    
    
"""

def train(pairs, labels, validation_data, destination, epochs=1000):

    lip_auth = LipAuth(weight_path='/Users/Carrie/git/LipNet/evaluation/models/unseen-weights178.h5')
  
    #callbacks = [ModelCheckpoint(checkpointFile, save_best_only=True, save_weights_only=True)]

    
    # setting loss to really big number so the start will be lower
    n_examples = len(labels)
    n_val_examples = len(validation_data[1])
    best_loss = np.inf
    EER_at_low_val = np.inf
    epoch_at_low_val = 0
    threshold_at_low_val = 0
    for i in range(epochs):
        print("\nEpoch %d/%d" %(i+1,epochs))
        bar = tqdm(range(nt_examples))
        loss = 0
        val_loss = 0
        for j in bar: # assumes batch size of 1
            bar.set_description('%d/%d'%(j,n_examples))
            loss += lip_auth.lipAuth.train_on_batch([np.array(pairs[0][j]), \
                np.array(pairs[1][j])], np.array([labels[j]]))
            bar.set_postfix(loss=loss/float(j+1))
        
        scores = []
        for k in range(n_val_examples):
            val_loss += lip_auth.lipAuth.test_on_batch([np.array(validation_data[0][0][k]), \
                                                        np.array(validation_data[0][1][k])], \
                                                       np.array([validation_data[1][k]]))
            pred = lip_auth.lipAuth.predict_on_batch([np.array(validation_data[0][0][k]), \
                                                        np.array(validation_data[0][1][k])])
            scores.append(pred[0][0])

        FPR, TPR, thresholds = roc_curve(np.array(validation_data[1]), np.array(scores))   
        EER = brentq(lambda x : 1. - x - interp1d(FPR, TPR)(x), 0., 1.)
        threshold = interp1d(FPR, thresholds)(EER)

                
                
        print('validation loss : %.3lf' % (val_loss/float(n_val_examples)))
        print("EER = ", EER)
        print("Threshold = ", threshold)
        
        
        if val_loss/float(n_val_examples) < best_loss:
            best_loss = val_loss/float(n_val_examples)
            print("validation loss decreased ... EER changed from : ", EER, " to : ", EER_at_low_val)
            
            EER_at_low_val = EER
            epoch_at_low_val = i+1
            threshold_at_low_val = threshold

            
            name = destination + "weights_for_lipAuthModel_EER_" + str(EER_at_low_val) + \
            "_threshold_" + str(threshold_at_low_val) + "_atEpch" + str(epoch_at_low_val) +".h5"
            lip_auth.lipAuth.save_weights(name)



    return EER_at_low_val, threshold_at_low_val, epoch_at_low_val


"""

def main():
    path = '/Users/Carrie/Desktop/xm2demo/mats_75/'
    destination = "/Users/Carrie/git/LipAuth/test_10ppl_retrainingAfterLayer14/"
    epochs = 100
    
    listallVids = ""
    
    
    best_EER, best_threshold, best_epoch = train(pairs, labels, validation_data, \
                                                 checkpointFile, destination, epochs=epochs)
    
    #best_EER, best_threshold, best_epoch = train([pairs[0][:20], pairs[0][:20]], \
    #                                             labels[:20], ([validation_data[0][0][:10],\
    #                                validation_data[0][1][:10]], validation_data[1][:10]),\
    #            checkpointFile, destination, epochs=3)
    
    print("Final Best EER: ", best_EER)
    print("Final Best Threshold: ", best_threshold)
    print("Final Best Epoch: ", best_epoch )
    
    
    """
    path = '/Users/Carrie/Desktop/xm2demo/mats_75/'
  
  # get training data
    pairs, labels = get_pairs(path, 'training.txt')
    
    
  
    # get validation data
    validation_data = get_pairs(path, 'validation.txt')
  
    checkpointFile = 'lipAuth_test_best_checkpoint.h5'
    destination = "/Users/Carrie/git/LipAuth/test_10ppl_retrainingAfterLayer14/"

    epochs = 100
    best_EER, best_threshold, best_epoch = train(pairs, labels, validation_data, \
                                                 checkpointFile, destination, epochs=epochs)
    
    #best_EER, best_threshold, best_epoch = train([pairs[0][:20], pairs[0][:20]], \
    #                                             labels[:20], ([validation_data[0][0][:10],\
    #                                validation_data[0][1][:10]], validation_data[1][:10]),\
    #            checkpointFile, destination, epochs=3)
    
    print("Final Best EER: ", best_EER)
    print("Final Best Threshold: ", best_threshold)
    print("Final Best Epoch: ", best_epoch )
    """

if __name__ == '__main__':
    main()
