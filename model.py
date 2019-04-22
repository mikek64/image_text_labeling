# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 08:07:08 2018

Model for processing data

@author: mike_k
"""
#%%%

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config
 # home brew module to replicate some of the keras functionailty with pytorch
from keraspytorch import CompiledModel
#%%  The neural net

class Net(nn.Module):
    ''' CNN to analyse the 'pseudo image' of the word vectorisations'''
    
    def __init__(self, input_channels, output_channels):
        # the input channels relate to the word vectorisation
        # the output channels relate to the number of different tags
        channels = 64 # internal number of channels
        self.numlayers = 20 
        
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channels,
                               kernel_size = 3, 
                               stride = 1, 
                               padding = 1)
        self.drop1 = nn.Dropout2d(p = 0.2)
        
        
        self.convlayers = nn.ModuleList(nn.Conv2d(channels, channels, 
                                                  kernel_size = 3, 
                                                  stride = 1, 
                                                  padding = 1)
                                        for i in range(self.numlayers))
        self.dropoutlayers = nn.ModuleList(nn.Dropout2d(p = 0.2)
                                            for i in range(self.numlayers))
          
        self.batchnorm = nn.ModuleList(nn.BatchNorm2d(channels)
                                        for i in range(self.numlayers))
        
        self.convlast = nn.Conv2d(channels, output_channels, 
                                  kernel_size = 3, 
                                  stride = 1, 
                                  padding = 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = self.drop1(x)
        
        for i in range(self.numlayers):     
            x = F.relu(self.batchnorm[i](self.convlayers[i](x)))
            #x = self.dropoutlayers[i](x) # dropout not improving results

        x = self.convlast(x)
        return x



#%%  # Loss function
    
class Loss():
    ''' Loss calculation 
        This is effectively a pixel level segmentation problem so we calculate
        the loss separately for each pixel in the 64 * 64 image and sum
        However we are only interested in those pixels where there are words
        embedded so we exclude those where no label is defined 
        (label 0 = 'Not a word') '''
    def __init__(self, output_size, device):
        ''' create an index tensor with one row (element) for each potential 
        output, containing the expected output values 0 to output_size.
        When combined with the truth (= label) tensor, this will produce a 
        4 grid of 1s and 0s representing the expected result as True/False
        The first entry is changed from 0 to -1 so no match (= loss) is 
        obtained from label 0 = 'Not a word'
        '''
        self.output_size = output_size
        
        self.ix_T = torch.zeros((1,output_size,1,1), 
                                dtype = torch.int64, device = device)
        for i in range(output_size):
            self.ix_T[0,i,0,0] = i
        self.ix_T[0,0,0,0] = -1

    def __call__(self, predict_T, truth_T):
        return self.loss(predict_T, truth_T)
    
    def loss(self, predict_T, truth_T):
        ''' Given prediction from the neural network, and truth image 
            calculate the loss.
            Prediction: Tensor, Size(Batch, Num_Classes, Height, Width)
            with value predicting the class, 1 channel for each class,
            but not softmaxed
            Truth: Tensor(Size(Batch, 1, Height, Width))
            with value one of the rgb classes.  
            If outside the class pixel is ignored '''
        # Assertions
        assert type(predict_T) == torch.Tensor
        assert type(truth_T) == torch.Tensor
        size_p = list(predict_T.size())
        size_t = list(truth_T.size())
        assert size_p[0] == size_t[0]
        assert size_p[2] == size_t[2]
        assert size_p[3] == size_t[3]
        assert size_p[1] == self.output_size
        assert size_t[1] == 1
        
        # calculate True/False (1,0) grid for class truths along channel axis
        true_T = truth_T == self.ix_T
        # Get log probabilities along channel dimension, i.e. for each class
        logp_T = F.log_softmax(predict_T, dim = 1)
        # calculate cross entropy loss, i.e. - q.log(p)
        loss_T = - logp_T[true_T].sum()
        return loss_T

#%%
   
     
def accy(x, y):
    ''' amended accuracy metric ignoring 0 entries
    x [Batch, output_size, Height, Width]
    y [Batch, 1, Height, Width]
    
    '''
    assert type(x) == torch.Tensor
    assert type(y) == torch.Tensor
    
    # remove single channel dimension
    y_max = y.squeeze(1)
    # Creates a mask for the non zero entries in the label as byte tensor 
    y_mask = y_max>0    
    # Get predicted entries
    x_max = x.max(1)[1]  #  argmax, i.e. max category along channel dimension
    
    # Restrict to non-zero entries (i.e. where there are words)
    y_max_masked = y_max[y_mask]
    x_max_masked = x_max[y_mask]
    
    count_all = np.prod(list(x_max_masked.size()))
    count_correct = (x_max_masked == y_max_masked).sum().item()
    acc = count_correct / count_all
      
    return acc  

#%%
def accy_plus(x, y):
    ''' amended accuracy metric ignoring 0 & 1 entries
    I.e. check it isn't just labelling everything unassigned
    x [Batch, output_size, Height, Width]
    y [Batch, 1, Height, Width]
    
    '''
    assert type(x) == torch.Tensor
    assert type(y) == torch.Tensor
    
    # remove single channel dimension
    y_max = y.squeeze(1)
    # Creates a mask for the non zero entries in the label as byte tensor 
    y_mask = y_max>1    
    # Get predicted entries
    x_max = x.max(1)[1]  #  argmax, i.e. max category along channel dimension
    
    # Restrict to non-zero entries (i.e. where there are words)
    y_max_masked = y_max[y_mask]
    x_max_masked = x_max[y_mask]
    
    count_all = np.prod(list(x_max_masked.size()))
    count_correct = (x_max_masked == y_max_masked).sum().item()
    acc = count_correct / count_all
      
    return acc

            
#%%
def new_model(input_size, output_size):
    ''' combine model and loss function into keras like functionality'''
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    net = Net(input_size, output_size)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters())
    lossfn = Loss(output_size, device)
    metrics = [accy, accy_plus]
    predictfn = nn.Softmax(dim = 1)
    
    compiled_model = CompiledModel(net, optimizer, lossfn, metrics,
                                   predictfn = predictfn)
    return compiled_model


    
    
    