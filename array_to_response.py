# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 08:11:23 2018

@author: mike_
Converts numpy arrays (invoice encoding and a word map) for a single file
to a list of lists with 
        [word, predicted tag index, predicted tag, probability]

"""

#%%
import config
import table_to_array
import ml
import numpy as np

#%%  MASTER FILE PROCESSING

# load label master to 2 dictionaries in both ways
with open(config.local_cfg['LABEL_MASTER'],'r') as f:
    t = f.read()

label_to_id = {}
id_to_label = {}

for x in t.split('\n')[1:]:  # ignore header
    num, lbl = x.split(',')
    label_to_id[lbl] = num
    id_to_label[num] = lbl


#%%

def get_model():
    ''' load the model '''
    compiled_model = ml.compile_model() # get model
    
    # Get saved weights
    model_file = config.local_cfg['MODEL']
    if model_file[-4:] != '.pkl': 
        # model file is a path: get latest compatible model
        name = ml.latest_compatible_model(model_file)
        model_file += '/' + name
    
    compiled_model.load(model_file) # load weights
    return compiled_model

trained_model = get_model() 

#%%

def predict_from_array(npa, wma, words):
    ''' predict an invoice contents from an array 
        returns a list of lists with 
        [word, predicted tag index, predicted tag, probability]'''  
    assert type(npa) == np.ndarray # encoded transformed data for one invoice
    assert type(wma) == np.ndarray # word map array
    assert type(words) == list  # list of the words from Vision
        
    npa = np.expand_dims(npa,0) # add batch dimension
    npa = ml.reshape_data(npa) # transpose axes for pytorch
    # predict tags using model
    prediction = trained_model.predict(npa) # detect labels
    # prediction is an array of probabilities for each pixel
    # [batch, label index, Height, Width]
    prediction = prediction.squeeze(0) # remove batch dimension
    
    # interpret response
    wm = np.expand_dims(wma,0) # so same number of axes as prediction [_, Height, Width]  
    response = []
    for i,w in enumerate(words):
        lst = []
        lst.append(w) # word being predicted

        # Average probabilities for each pixel in word map for that word
        wm_mask = wm == i # Mask of cells for this item
        n_cells = np.sum(wm_mask) # number of cells for this item
        if n_cells != 0:
            probs_masked_cells = prediction * wm_mask # probabilities for these cells only
            probs_item = np.sum(probs_masked_cells, (1,2)) / n_cells # average cells probabilities
            pred = np.argmax(probs_item) # prediction is item with max probability
            prob = np.max(probs_item)  # probability of prediction
            tag = id_to_label[str(pred)]
        else:
            # word not in the word map, probably because of location overlap
            pred = -1
            prob = 0
            tag = 'Not in map'            
        
        lst.append(pred) # index of predicted tag
        lst.append(tag) # label associated with tag
        lst.append(prob) # probability that prediction is correct
        response.append(lst)
    
    return response

if __name__ == '__main__':
    # Test
    TEST_FILE = config.local_cfg['TEST_FILE']
    npa = np.load(config.patharray + '/' + TEST_FILE + '_data.npy') # load array
    wma = np.load(config.pathwordmap + '/' + TEST_FILE + '_wordmap.npy') # load word map
    words, _ = table_to_array.read_table(TEST_FILE, config.pathtable) # load words as a list
    response = predict_from_array(npa, wma, words)
    s = '\n'.join(['\t'.join([str(y) for y in x]) for x in response])
    print(s)