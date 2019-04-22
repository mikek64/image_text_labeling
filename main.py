# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:15:17 2018

Read an invoice image and identify labels
This is the main program that combines all components

@author: mike_k
"""
#%%
import os
import config
import image_to_base64
import base64_to_json
import json_to_table
import table_to_array
import array_to_response
import numpy as np

#%%

def call_vision(path_to, path_from, file, ext = '.png', verbose = True):
    ''' converts an invoice image to an array as preparation to submit
    to model'''
    
    if verbose: print('Converting image to base 64 ...')
    image_to_base64.convert_file(path_from, 
                                 path_to, file, ext)
    if verbose: print('Reading image as words with Google Vision ...')
    base64_to_json.read_image(path_to, 
                              path_to, file)
    

def predict_from_json(path, file, verbose = True):
    ''' predict invoice contents from json
        returns a list of lists with 
        [word, predicted tag index, predicted tag, probability]'''    
    if verbose: print('Converting response json to table ...')  
    jsn = json_to_table.read_json(file, path)  
    words, location = json_to_table.get_words_and_locations(jsn)
    
    if verbose: print('Vectorising words and encoding with location as numpy arrays ...')    
    ay, wmap_ay, _ = table_to_array.make_arrays(words,location)
    
    if verbose: print('Interpreting arrays ...')    
    response = array_to_response.predict_from_array(ay, wmap_ay, words)
    
    return response


def predict_from_array(file):
    ''' predict an invoice contents from an array from the training location
        returns a list of lists with 
        [word, predicted tag index, predicted tag, probability]
        '''  
    # load invoice array    
    ay = np.load(config.patharray + '/' + file + '_data.npy') # load array
    wmap_ay = np.load(config.pathwordmap + '/' + file + '_wordmap.npy') # load word map
    words, _ = table_to_array.read_table(file, config.pathtable) # load words as a list
    response = array_to_response.predict_from_array(ay, wmap_ay, words)
    
    return response
 
#%% save output
    
def save_response(path, file, response):
    ''' save response to file as tab delimited'''
    s = 'word\ttag_index\ttag\tprobability\n'
    s += '\n'.join(['\t'.join([str(y) for y in x]) for x in response])

    # Save as csv
    with open(path + '/' + file + '_response.txt','w', encoding = 'utf-8') as f:
        f.write(s)

#%%

def split_file(f):
    ''' splits file into path, file and ext '''
    path, file = os.path.split(f)
    file, ext = os.path.splitext(file)
    # defaults
    if path == '': 
        path = config.pathimages
    if ext == '':
        ext = '.png'
    return path, file, ext
    
if __name__ == '__main__':
    assert split_file('inv') == (config.pathimages, 'inv', '.png')
    assert split_file(r'C:\users\mike\inv.jpeg') == (r'C:\users\mike', 
                                                     'inv', '.jpeg')

#%% Get non-unassigned responses
    
def assigned_responses(response):
    ''' create a list of the non unassigned responses.
    Principally to give a cut down list of detections to illustrate output
    response is list of 
    [word, predicted tag index, predicted tag, probability]
    returns list of [tag, [words]] in tag index order'''
    wordd = {}
    usedtags = {}
    for r in response:
        ix = r[1] # index of label
        if ix > 1: # ie neither unassigned or not a word
            if ix in wordd:
                wordd[ix].append(r[0]) # add detected word
            else:
                usedtags[ix] = r[2] # record tag
                wordd[ix] = [r[0]] # start a list
    
    # loop through detections in ascending tag order
    assigned = []
    for ix in sorted(list(usedtags.keys())):
        assigned.append([usedtags[ix], wordd[ix]])
        
    return assigned

#%%


def predict_an_invoice(file, verbose = True, 
                       save = False, path = config.pathresponse,
                       print_response = False, 
                       print_assigned = False):
    ''' Predict invoice, either from image, or from array
    returns list of lists
    [word, predicted tag index, predicted tag, probability]'''
    path_from, file, ext = split_file(file)
    
    call_vision(config.local_cfg['TEMP_DATA'], path_from,
                file, ext, verbose)  # get json from Google Vision
    response = predict_from_json(config.local_cfg['TEMP_DATA'], file, verbose)
    
    if print_response: 
        print('All Responses and probabilities')
        for r in response:
            print(r) 
    
    if print_assigned:
        detections = assigned_responses(response)
        print('\nSpecific detections')
        for d in detections:
            print(d[0],' '.join(d[1]))
    if save:
        if path is None:
            path = config.pathresponse
        save_response(path, file, response)
    return response


if __name__ == '__main__':
    # testing
    response = predict_an_invoice(config.local_cfg['TEST_FILE'], save = True,
                                  print_response = True,
                                  print_assigned = True)

#%%
    
if __name__ == '__main__':
    # testing
    predict_an_invoice(config.local_cfg['TEST_FILE'])


    
    
