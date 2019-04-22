# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:18:33 2018

Reads a base64 image file and returns output as json using Google Vision

Submits the request to Vision using curl downloaded from https://curl.haxx.se/windows/
see https://cloud.google.com/vision/docs/using-curl
example call:
curl -v -s -H "Content-Type: application/json"  https://vision.googleapis.com/v1/images:annotate?key=APIKEY  --data-binary @request.json --output example_return.json

Needs an  api key for authenticating to Cloud Vision.  
See https://cloud.google.com/vision/docs/auth#using_an_api_key

Relatively slow: processed 12 files per minute

@author: mike_k
"""
#%%
import subprocess
import os
from sys import stdout
import config

#%%

# temp file for saving JSON to prior to submitting
request_json_file = config.local_cfg['TEMP_DATA'] + '/request.json'
path_from = config.pathbase64
path_to = config.pathjson

#%%  Load apikey for authorisation

with open(config.local_cfg['KEYFILE'], 'r') as f:
    apikey = f.read()

if __name__ == '__main__':
    print(apikey)

#%%  Example request JSON from a web image for use in testing

REQUEST_JSON_WEB_IMAGE = '''
{
  "requests": [
    {
      "image": {
        "source": {
          "imageUri": "https://cloud.google.com/vision/images/rushmore.jpg"
        }
      },
      "features": [
        {
          "type": "LANDMARK_DETECTION",
          "maxResults": 1
        },
        {
          "type": "WEB_DETECTION",
          "maxResults": 2
        }
      ]
    }
  ]
}'''
if __name__ == '__main__':
    print(REQUEST_JSON_WEB_IMAGE)

#%% Read image file

def read_image_file(pathfrom, file):
    ''' read the image which is in base 64 format '''
    with open(pathfrom + '/' + file + '.txt', 'r') as f:
        s = f.read()
    return s
    
if __name__ == '__main__':
    image_b64 = read_image_file(path_from, config.local_cfg['TEST_FILE'])
    print(image_b64)
    
   
#%%  Our JSON

def create_json(base64):
    ''' create json and write to file '''
    jsn = '''
{
  "requests": [
    {
      "image": {
        "content": "''' + base64 + '''" 
      },
      "features": [
        {
          "type": "TEXT_DETECTION"
        }
      ]
    }
  ]
}'''

    # write our request_json to a file for processing
    #  can not submit it direct in curl as base64 encoding is too long
    with open(request_json_file, 'w') as f:
        f.write(jsn)

    return jsn

if __name__ == '__main__':
    request_json = create_json(image_b64)
    print(request_json)


#%%

## QUESTION:  should we be focing the charset=utf-8 ?
# --tlsv1.2 foces it to use tlsv1.2 or greater.  Vision api appears not to support tlsv1.3

def call_vision(pathto, file):
    subprocess.call([config.local_cfg['CURL'],
                     '-s',
                     '--tlsv1.2',
                     '-H',    
                     "Content-Type: application/json; charset=utf-8",
                     "https://vision.googleapis.com/v1/images:annotate?key=" + apikey,
                     "--data-binary",
                     "@" + request_json_file,
                     "--output",
                     pathto + '/' + file + '.json'])


if __name__ == '__main__':
    call_vision(path_to, config.local_cfg['TEST_FILE'])
    
    
#%%
def read_image(pathfrom, pathto, file):
    ''' Read base64 image file to json using Google Vision '''
    image_b64 = read_image_file(pathfrom, file)
    request_json = create_json(image_b64)
    call_vision(pathto, file)   
    
    
#%%
      
def read_all_images(pathfrom, pathto):
    ''' Read all base64 image files in a folder from Google Vision '''
    tree = os.walk(pathfrom)
    files = list(tree)[0][2]
    n = len(files)   
    for i,f in enumerate(files):
        fbase = f[:-4] # strip the .txt from the file
        if config.OVERWRITE or not os.path.isfile(pathto + '/' + fbase + '.json'):
            read_image(pathfrom, pathto, fbase)
        stdout.write('\r' + str(i+1) + '/' + str(n)) # show progress
    stdout.write('')
    
if __name__ == '__main__':
    read_all_images(path_from, path_to)