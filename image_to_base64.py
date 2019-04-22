# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 2018

@author: mike_k

Convert image files to base64 to be in suitable format for sending to 
Google Cloud Vision api
Uses b64.exe downloaded from https://sourceforge.net/projects/base64/
call with   b64.exe -e input.jpg  output.txt
"""
#%%
import subprocess
import os
from sys import stdout
import config

#%%
def convert_file(pathfrom, pathto, file, ext = '.png'):
    subprocess.call([config.local_cfg['B64'], 
                     '-e', 
                     pathfrom + '/' + file + ext,
                     pathto + '/' + file + '.txt'])
    
if __name__ == '__main__':
    convert_file(config.pathimage, config.pathbase64,
                 config.local_cfg['TEST_FILE'])

#%%

def convert_all_files(pathfrom, pathto):
    tree = os.walk(pathfrom)
    files = list(tree)[0][2]
    n = len(files)   
    for i,f in enumerate(files):
        fbase = f[:-4] # strip the .png from the file
        convert_file(pathfrom, pathto, fbase)
        stdout.write('\r' + str(i+1) + '/' + str(n)) # show progress
    stdout.write('')
    
if __name__ == '__main__':
    convert_all_files(config.pathimage, config.pathbase64)


        


