# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 09:24:53 2018

@author: mike_

Original training data generation helper for when training data comprises
raw invoice images.  Generating the training data directly as word
location text files is easier if trying to generate variation as in
generate_training_data file

Functions to call data preparation for training to take data from
images through to arrays

"""

import config
import image_to_base64
import base64_to_json
import json_to_table
import table_to_array

#%%

''' Convert images to base 64 to the correct format for submision 
to Google Vision '''

if __name__ == '__main__':
    pathimg = config.pathimage
    if config.local_cfg.get('IMAGE_SUBFOLDER') is not None:
        pathimg += '/' + config.local_cfg.get('IMAGE_SUBFOLDER')
    
    image_to_base64.convert_all_files(pathimg, config.pathbase64)

#%%  

''' Submit to Google Vision and get json in return 
Submission via curl - security tlsv1.2'''

if __name__ == '__main__':
    base64_to_json.read_all_images(config.pathbase64, config.pathjson)
    
#%%  

''' Convert json to tables '''

if __name__ == '__main__':
    json_to_table.create_all_tables(config.pathjson, config.pathtable)
    

#%% 
'''Create labels (as list in csv): 
This is specific to the type of image being processed
E.g. original test labels were done in Excel tomatch with the words.'''
    
#%%  # create labels before doing this
    
if __name__ == '__main__':
    table_to_array.create_all_arrays(config.pathtable,
                                     config.pathlabel,
                                     config.pathwordmap,
                                     config.patharray,
                                     config.pathlabelarray)
    
#%%
''' After creating arrays manually split data between test and train
by moving from patharray to train and test subfolders '''
    