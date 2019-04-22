# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:13:04 2018

@author: mike_k

Read Cloud Vision Google JSON 
Extracts the words and locations and saves them to a file
This file is much smaller than the original JSON and hence
this facilitates transfer of the data

"""
#%%
import json
import numpy as np
import os
from sys import stdout
import config


#%%

# vertex order in location file as defined from Google Vision
TOPLEFTX = 0
TOPLEFTY = 1
TOPRIGHTX = 2
TOPRIGHTY = 3
BOTTOMRIGHTX = 4
BOTTOMRIGHTY = 5
BOTTOMLEFTX = 6
BOTTOMLEFTY = 7

#%%
path_from = config.pathjson
path_to = config.pathtable

#%% INDIVIDUAL DATA FILE PROCESSING

def read_json(file_name, path):
    with open(path + '/' + file_name + '.json','r', encoding='utf-8') as f:
        t = f.read()
    return json.loads(t)

if __name__ == '__main__':
    jsn = read_json(config.local_cfg['TEST_FILE'], path_from)
    
#%%

# Load descriptions and text to table

def get_words_and_locations(jsn):
    ''' Gets words and locations from JSON file '''
    # Get words
    n = len(jsn["responses"][0]["textAnnotations"])
    words = [jsn["responses"][0]["textAnnotations"][x]["description"] 
            for x in range(1,n)]
    
    # Get locations
    location = np.zeros((n-1,8))
    for i in range(n-1):
        for j in range(4):
            location[i,j*2] = jsn["responses"][0]["textAnnotations"][i+1]["boundingPoly"]["vertices"][j]["x"]
            location[i,j*2+1] = jsn["responses"][0]["textAnnotations"][i+1]["boundingPoly"]["vertices"][j]["y"]
    
    return words, location

if __name__ == '__main__':
    words, location = get_words_and_locations(jsn)

#%% save the word table

def save_word_table(file, path, words, location):
    # words and locations as tab delimited txt file
    s = 'text\tx-top-left\ty-top-left\tx-top-right\ty-top-right\tx-bottom-right\ty-bottom-right\tx-bottom-left\ty-bottom-left'
    n = len(words)
    for i in range(n):
        s += '\n' + words[i]
        for j in range(8):
            s += '\t' + str(location[i,j])
    
    # Save as csv
    with open(path + '/' + file + '_table.txt','w', encoding = 'utf-8') as f:
        f.write(s)

if __name__ == '__main__':
    save_word_table(config.local_cfg['TEST_FILE'], path_to, words, location)

#%%
    
def create_table(pathjson, pathtable, file):
    ''' create tablefile for a single file '''
    jsn = read_json(file, pathjson)   
    words, location = get_words_and_locations(jsn) 
    save_word_table(file, pathtable, words, location)


if __name__ == '__main__':
    create_table(path_from, path_to, config.local_cfg['TEST_FILE'])


#%% process all files in PATH

def create_all_tables(pathfrom, pathto):
    tree = os.walk(pathfrom)
    files = list(tree)[0][2]
    n = len(files)
    # strip the .json off the file name
    for i,f in enumerate(files):
        if f[-5:] == '.json':
            # Strip .json and submit
            create_table(pathfrom, pathto, f[:-5])
            stdout.write('\r' + str(i+1) + '/' + str(n)) # show progress
    stdout.write('')

if __name__ == '__main__':
    create_all_tables(path_from, path_to)
    print('Finished')

