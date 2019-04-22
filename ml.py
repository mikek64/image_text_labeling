# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 08:08:33 2018

Machine Learning to train model.  Run each cell separately in order


Model naming convention: Model_Version_inputxoutput_YYYY-MM-DD-HH-MM-SS
e.g. Model_3-1_56x26_2018-12-02-12-07-26
Version and input & output size must match for valid model

@author: mike_k
"""
#%%
import numpy as np
import config
import os
from sys import stdout
import datetime
from matplotlib import pyplot as plt

from word_to_vector import EMBED_SIZE as INPUT_SIZE 
from table_to_array import TAG_SIZE as OUTPUT_SIZE
import model

#%% build train and test sets

''' Data must be manually copied to the train & test folders. 
'''

def load_nparray(fullfile):
    ''' load an array and add a dimension'''
    npa = np.load(fullfile)
    npa = np.expand_dims(npa,0)
    return npa    


def load_arrays(pathdata, pathlabel):
    ''' load arrays to single arrays to generate train and test sets 
        generates data, label  arrays'''
    tree = os.walk(pathdata)
    files = list(tree)[0][2]
    data = []
    label = []
    n = len(files)
    for i,f in enumerate(files):
        data.append(load_nparray(pathdata + '/' + f))
        f_label = f[:-9] + '_labels.npy'
        label.append(load_nparray(pathlabel + '/' + f_label))
        stdout.write('\r' + str(i+1) + '/' + str(n)) # show progress
    stdout.write('\n') 
    
    return np.concatenate(data), np.concatenate(label)


if __name__ == '__main__':
    ''' get train and test data '''
    np_train_data, np_train_labels = load_arrays(config.pathtrain, 
                                                 config.pathlabelarray)
    np_test_data, np_test_labels = load_arrays(config.pathtest, 
                                               config.pathlabelarray)
    print(np_train_data.shape, np_train_labels.shape,
          np_test_data.shape, np_test_labels.shape)
        
#%% Reshape train data

def reshape_data(npa):
    ''' Reshape [Batch, Height, Width, Channel] to
        [Batch, Channel, Height, Width] for pytorch '''
    return np.transpose(npa, (0,3,1,2))

def expand_labels(npa):
    ''' labels have shape [Batch, Height, Width] and need
         a channel dimension [Batch, Channel, Height, Width] '''
    if len(npa.shape) == 3:
        npa = np.expand_dims(npa,1) 
    return npa

if __name__ == '__main__':
    if len(np_train_labels.shape) == 3:  # prevent running twice by accident
        np_train_data = reshape_data(np_train_data)
        np_test_data = reshape_data(np_test_data)
        np_train_labels = expand_labels(np_train_labels)
        np_test_labels = expand_labels(np_test_labels)
        print(np_train_data.shape, np_train_labels.shape,
              np_test_data.shape, np_test_labels.shape)
    
    
#%%  Build new model

def compile_model():
    return model.new_model(INPUT_SIZE, OUTPUT_SIZE)

if __name__ == '__main__':
    compiled_model = compile_model()


#%% load  Run this to load a previous model for further training

def latest_compatible_model(path):
    ''' returns the latest model name with the VERSION and Input and Output
    sizes compatible '''
    version = 'v' + config.VERSION
    sizes = str(INPUT_SIZE) + 'x' + str(OUTPUT_SIZE)
    # Walk through
    tree = os.walk(path)
    files = list(tree)[0][2]
    latest_name = ''
    latest_index = -1
    for i,f in enumerate(files):
        ext = f[-4:]
        parts = f[:-4].split('_')
        if len(parts) != 4 or ext != '.pkl':
            continue
        if (parts[0] != 'Model' or 
            parts[1] != version or
            parts[2] != sizes):
            continue
        if parts[3] > latest_name:
            latest_index = i # most recent version
            
    assert latest_index != -1, 'No compatible version found in ' + path
        
    return files[latest_index]
    
def load_model(model, file):
    model.load(config.local_cfg['MODEL'] + '/' + file)

if __name__ == '__main__':
    filename = latest_compatible_model(config.pathmodels)
    load_model(compiled_model, filename)
    print(filename)

#%%   train
''' 
With initial training set of 1000 arrays  100 epochs were sufficient
Much larger training sets caused out of memory issue on my GPU.  Next cell
contains batch train option to get round this

Progress Metrics:
accy is amended accuracy metric ignoring 0 labels, i.e. blank spaces
accy plus is amended accuracy metric ignoring 0 & 1 entries (blank & unassigned)
    I.e. check it isn't just labelling everything unassigned '''


def train(epochs):
    return compiled_model.fit(np_train_data, np_train_labels, epochs = epochs, 
                              batch_size = 32) #, validation_split = 0.2)

if __name__ == '__main__':
    hist = train(10)


#%%  Batch train
''' Train in major batches.  Use to split data into parts to avoid running
out of GPU memory '''

# number of items to load as tensors from which batches are selected
LARGE_BATCH_SIZE = 1000

def batch_train(num_large_batches, epochs):
    for i in range(num_large_batches):
        print('Batch ' + str(i) + '/' + str(num_large_batches))
        sample = np.random.choice(np_train_data.shape[0], 
                                  size = LARGE_BATCH_SIZE, 
                                  replace = False)
        compiled_model.fit(np_train_data[sample], 
                           np_train_labels[sample], 
                           epochs = epochs, 
                           batch_size = 32)

if __name__ == '__main__':
    batch_train(num_large_batches = 50, epochs = 5) 

#%% save

def generate_model_name():
    ''' generate new model name in standard format '''
    name = 'Model'
    name += '_v' + config.VERSION
    name += '_' + str(INPUT_SIZE) + 'x' + str(OUTPUT_SIZE)
    name += '_' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    name += '.pkl'
    return name
   
def save_model(model, file):
    model.save(config.local_cfg['MODEL'] + '/' + file)
    
if __name__ == '__main__':
    filename = generate_model_name()
    print(filename)     
    save_model(compiled_model, filename)
    

    
#%%  Chart training history.  Will not work with Batch train

if __name__ == '__main__': 
    n_epochs = len(hist['accy_plus'])
    epochs = list(range(n_epochs))
    
    plt.plot(epochs, hist['accy_plus'], 'b', label = 'train')
    plt.plot(epochs, hist['val_accy_plus'], 'r', label = 'val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training')
    plt.show()
    
#%% test data

if __name__ == '__main__':
    # test_metrics are loss, accy, accy_plus
    test_metrics = compiled_model.evaluate(np_test_data,
                                           np_test_labels, 
                                           batch_size = 32)
    print('Test accuracy: {:.4f}'.format(test_metrics[1]),
          'Test accuracy plus: {:.4f}'.format(test_metrics[2]))
    
#%%  Predictions
if __name__ == '__main__':    
    predictions = compiled_model.predict(np_test_data)
    print(predictions.shape)  #[Batch, output_size, Height, Width]
    
#%% Ascertain errors for investigation

def incorrect_plus(x,y):
    ''' reverse f the accuracy plus metric
    given
    x [Batch, output_size, Height, Width]
    y [Batch, 1, Height, Width]
    returns an array with 1's of each incorrect label element identified
    Use this to identify classification errors for subsequent investigation 
    
    NB although superficially similar to accy_plus this accepts arrays as 
    input
    '''
    assert type(x) == np.ndarray
    assert type(y) == np.ndarray
    
    # remove single channel dimension
    y_max = y.squeeze(1)
    # Creates a mask for the non zero entries in the label as byte tensor 
    y_mask = y_max>1

    # Get predicted entries
    x_max = np.argmax(x,1)  #  i.e. max category along channel dimension
    
    # Restrict values to non-zero entries (i.e. where there are words)
    # create a grid of items
    y_max_masked = y_max * y_mask
    x_max_masked = x_max * y_mask
    
    incorrect = x_max_masked != y_max_masked
      
    return incorrect 


if __name__ == '__main__': 
    errors = incorrect_plus(predictions, np_test_labels)
    x_errors = np.sum(errors, (1,2)) # sum across all but batch axis
    print('Total number of errors = ', x_errors.sum())
    all_errors = [(i,x) for i,x in enumerate(x_errors) if x > 0]
    print('Error items index and number of errors')
    print(all_errors)


#%% Issue investigation

if __name__ == '__main__':
    # testing  Compare labels with detections
    import table_to_array
    import main
    
    TEST_FILE = ''
    response = main.predict_from_array(TEST_FILE)
    labels = table_to_array.load_labels(TEST_FILE, config.pathlabel)
    for i, r in enumerate(response):
        if r[2] != labels[i]:
            e = '**issue** ' + str(r[3])
        else:
            e = ''
        print(r[0], '\t', r[2], '\t', labels[i], '\t', e)
