# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:13:04 2018

@author: mike_k

Read Cloud Vision Google the word table representation
and text list of labels
Processes them as:-
    words -  a list of all words identified in the JSON
    location - a np array of the x,y boundary box as defined by the json
    labels - a list of the word labels
    position - np array defining the location as remapped to a GRID_SIZE array
Generates numpy arrays for data and labels in suitable format for training

"""
#%%

import numpy as np
import os
from sys import stdout
import config
from read_wordtable import read_wordtable_to_words_and_locations as read_table
from location_to_grid import get_map, GRID_SIZE
from word_to_vector import embed_words, EMBED_SIZE

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

TAG_SIZE = len(id_to_label) 
# TAG_SIZE also defines the output size for the neural network

if __name__ == '__main__':
    print(TAG_SIZE)

#%% INDIVIDUAL DATA FILE PROCESSING

#%% save array file
    
def save_array(file, path, array):
    np.save(path + '/' + file, array)

#%%  Read word table for testing
if __name__ == '__main__': 
    words, location = read_table(config.local_cfg['TEST_FILE'], 
                                 config.pathtable)

    wordsembed = embed_words(words) # numpy array for word embedding   
    wma = get_map(location)  # snap location to grid
    save_array(config.local_cfg['TEST_FILE'] + '_wordmap.npy', 
               config.pathwordmap, wma)

#%%

def create_data_array2(words, map_array, wordsembed):
    ''' Creates data as numpy array '''
    ay = np.zeros((GRID_SIZE,GRID_SIZE,EMBED_SIZE), dtype = np.float32) 
    n = len(words)
    for i in range(n):  # Loop through each word
        ay[map_array == i] = wordsembed[i]
    return ay


if __name__ == '__main__':
#    ay = create_data_array(words, position, wordsembed)
    ay2 = create_data_array2(words, wma, wordsembed)

#%%

def load_labels(file, path): 
    with open(path + '/' + file + '_labels.csv','r', encoding = 'utf-8') as f:
        t = f.read()
    labels = t.split('\n')[1:]  # exclude the first header row
    return labels

if __name__ == '__main__':
    labels = load_labels(config.local_cfg['TEST_FILE'], config.pathlabel)
    assert len(labels) == len(words)  # labels should have same length as words
    print(labels[0:5])

#%%

def create_labels_array2(map_array, labels):
    ''' Create data label array'''
    # int64 for pytorch consumption
    target = np.zeros((GRID_SIZE,GRID_SIZE), dtype = np.int64) 
    
    n = len(labels)
    for i in range(n):  # Loop through each word
        target[map_array == i] = int(label_to_id[labels[i]])
    return target

if __name__ == '__main__':
    target = create_labels_array2(wma, labels)
    save_array(config.local_cfg['TEST_FILE'] + '_labels.npy', 
               config.pathlabelarray, target)

#%%
    
def make_arrays(words, location, labels = None):
    ''' make the arrays for use in the machine learning '''
    wmap_ay = get_map(location) # translate location to GRID_SIZE grid  
    wordsembed = embed_words(words) # vectorize the words
    ay = create_data_array2(words, wmap_ay, wordsembed)

    if labels is not None:
        label_ay = create_labels_array2(wmap_ay, labels)
    else:
        label_ay = None
    return ay, wmap_ay, label_ay        


#%% create and save arrays - used for processing many files
    
def create_arrays(pathtable, pathlabel, pathwordmap, pathto, 
                  pathlabelarray, file):
    ''' encode data and labels if they exist for a single file '''
    # Set up
    file_labels_np = file + '_labels.npy'
    file_data_np = file + '_data.npy'
    file_wordmap_np = file + '_wordmap.npy'

    # load from file
    words, location = read_table(file, pathtable)
    try: # Process labels if they exist
        labels = load_labels(file, pathlabel)
    except:
        labels = None

    ay, wmap_ay, label_ay = make_arrays(words, location, labels)

    save_array(file_data_np, pathto, ay)
    save_array(file_wordmap_np, pathwordmap, wmap_ay)
    if labels is not None:
        save_array(file_labels_np, pathlabelarray, label_ay)
        
    return ay, wmap_ay, label_ay

if __name__ == '__main__':
    create_arrays(config.pathtable, config.pathlabel, config.pathwordmap,
                  config.patharray, config.pathlabelarray,
                  config.local_cfg['TEST_FILE'])

#%% process all files in PATH

def create_all_arrays(pathtable, pathlabel, pathwordmap, 
                      pathto, pathlabelarray):    
    tree = os.walk(pathtable)
    files = list(tree)[0][2]
    n = len(files)
    for i,f in enumerate(files):
        create_arrays(pathtable,
                      pathlabel,
                      pathwordmap,
                      pathto,
                      pathlabelarray,
                      f[:-10]  # strip '_table.txt'
                      )
        stdout.write('\r' + str(i+1) + '/' + str(n)) # show progress
    stdout.write('\n')

if __name__ == '__main__':
    create_all_arrays(config.pathtable, config.pathlabel, 
                      config.pathwordmap, config.patharray,
                      config.pathlabelarray)
    print('\nFinished')

#%%  save labels and word map to csv for investigation
    
def save_to_text(file, path, array):  
    ''' saves a numpy array as a text file '''
    np.savetxt(path + '/' + file, array, delimiter = ',')

def save_arrays_to_csv(path, file):
    ''' save label and word map arrays to csv '''
    _, wordmap_np, label_np = create_arrays(config.pathtable, config.pathlabel,
            config.pathwordmap, config.patharray, config.pathlabelarray, file)    
    save_to_text(file + '_label_np.csv', path, label_np)
    save_to_text(file + '_wordmap_np.csv', path, wordmap_np)
    
if __name__ == '__main__':
    for f in ['inv_5_2','inv_5_3','inv_5_7','inv_5_9','inv_5_19']:
        # One invoice of each type
        create_arrays(config.pathtable, config.pathlabel, config.pathwordmap,
                      config.patharray, config.pathlabelarray,
                      f)
        save_arrays_to_csv(config.pathtxtarray, f)



#%% examine some files created
if __name__ == '__main__':
    a = np.load(r'C:\Users\009091\Documents\Repos\invoices\transformed\inv_5_10_labels.npy')
    b = np.load(r'C:\Users\009091\Documents\Repos\invoices\transformed\inv_5_9_labels.npy')
    c = np.load(r'C:\Users\009091\Documents\Repos\invoices\transformed\inv_5_10_data.npy')
    d = np.load(r'C:\Users\009091\Documents\Repos\invoices\transformed\inv_5_9_data.npy')
    e = np.load(r'C:\Users\009091\Documents\Repos\invoices\wordmap\inv_5_10_wordmap.npy')
