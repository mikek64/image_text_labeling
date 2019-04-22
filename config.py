# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 08:46:49 2018

@author: mike_k

Contains various config for the invoice processing module

Local config as to where working directories are on your machine should be 
entered as a tab delimited list in local_config.txt which is ignored by git
Create this by copying local_config_default.txt


"""
#%%
import os

#%%

''' Version number specifies the model vand the encoding.  It should be updated
whenever there are changes to: model.py, location_to_grid, word_to_vector '''
VERSION = '3-2'

#%%  Load config from config files

local_cfg = {} # dictionary of where local path

def add_config(config_file):
    ''' Adds config from the relevant file to local_cfg '''
    try:
        with open(config_file, 'r') as f:
            s = f.read()
        for line in s.split('\n'):
            rw = line.split('\t')
            if len(rw) == 2:
                local_cfg[rw[0]] = rw[1]
    except:
        pass
        
add_config('./local_config_default.txt')
add_config('./local_config.txt')

if __name__ == '__main__': 
    print(local_cfg.items())


#%% pathbase is where files will be stored for the training data
    
pathbase = local_cfg['TRAINING_DATA']

pathimage = pathbase + '/' + 'images' # original training images
pathbase64 = pathbase + '/' + 'imagesbase64' # images in base64
pathjson = pathbase + '/' + 'json' # json of words from Google Vision
pathtable = pathbase + '/' + 'tabledata' # for json info as table
pathlabel = pathbase + '/' + 'labels' # labels for the words
patharray = pathbase + '/' + 'array_data' # transformed data & as numpy
pathlabelarray = pathbase + '/' + 'array_labels' # transformed labels as numpy
pathwordmap = pathbase + '/' + 'array_wordmap' # array to word location
pathresponse = pathbase + '/' + 'response' # where the response is saved
pathtxtarray = pathbase + '/' + 'txtarrays' # save word map as csv for analysis

# Note files must be manually copied from path array to train and test subfolders
pathtrain = patharray + '/' + 'train' # training data set
pathtest = patharray + '/' + 'test' # test data set


pathmaster = "./masterdata"
pathmodels = local_cfg['MODEL']
pathimages = local_cfg['IMAGES'] # default location for images for evaluation

OVERWRITE = False # Overwrite existing files on batch processing

#%% # Make sub folders for training.  If new implemtation

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def make_folders():
    make_folder(pathbase64)
    make_folder(pathimage)
    make_folder(pathjson)
    make_folder(pathtable)
    make_folder(pathlabel)
    make_folder(patharray)
    make_folder(pathlabelarray)
    make_folder(pathwordmap)
    make_folder(pathresponse)
    make_folder(pathtrain)
    make_folder(pathtest) 
    make_folder(pathtxtarray)
if __name__ == '__main__':
    make_folders()
    


#%% Check components exist
if __name__ == '__main__':
    assert os.path.isfile(local_cfg['B64']), 'Missing: ' + local_cfg['B64'] + ' check config'
    assert os.path.isfile(local_cfg['CURL']), 'Missing: ' + local_cfg['CURL'] + ' check config'
    assert os.path.isfile(local_cfg['KEYFILE']), 'Missing: ' + local_cfg['KEYFILE'] + ' check config'    
