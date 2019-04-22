# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 09:48:09 2018
Implements keras like fit, evaluate and predict functionality for PyTorch models.
implement with:
compiled_model = CompiledModel(model, optimizer, lossfn, metrics)
training_history = compiled_model.fit(x_train, y_train)

[Transfers entire training arrays to Tensors upfront which is convenient
but for large train sets can causes memory issues on GPU.  Option for
transferring only for batch being trained would be helpful.]

@author: mike_
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest
import random
from timeit import default_timer as timer
from sys import stdout

#%%
class CompiledModel():
    '''Link model to optimizer, loss function and 
    metrics to build keras type ease of use with pytorch models '''
    def __init__(self, model, optimizer, lossfn, metrics,
                 predictfn = None):
        assert model.__class__.__bases__[0].__name__ == 'Module'
        assert type(metrics) == list

        self.model = model
        self.optimizer = optimizer
        self.lossfn = lossfn
        self.metrics = metrics
        self.predictfn = predictfn # additional function for predict, e.g. softmax
    
    
    def _set_training_mode(self, training_mode):
        ''' sets the model mode to training or evalation as appropriate '''
        self.training_mode = self.model.training
        if training_mode and not self.model.training:
            self.model.train()
        elif not training_mode and self.model.training:
            self.model.eval()
    
    def _reset_training_mode(self):
        ''' reset the training mode to what it was previously '''
        self._set_training_mode(self.training_mode)

    def device(self):
        ''' returns the device of the model, from parameters '''
        return next(self.model.parameters()).data.device.type
    
    def _set_data_type_to_tensor(self, data):
        ''' converts input data type to tensor if necessary 
        Can we force the numpy dtypes to be correct for torch?
        torch seems to expect float32 or int64'''
        if type(data) == np.ndarray:
            assert data.dtype in [np.float32, np.int64], 'PyTorch normally expects data as float32 or int64'
            data = torch.from_numpy(data).to(self.device())
            self.source_data_type = np.ndarray
        elif type(data) == torch.Tensor: 
            self.source_data_type = torch.Tensor
        elif type(data) in [list, tuple]:
            # unpack list recursively and convert each element
            data = [self._set_data_type_to_tensor(x) for x in data]
        else:
            assert False, 'Unknown data type: not numpy or torch tensor' 
        return data


    def _reset_data_type(self, data):
        ''' converts data back to originaldtype '''
        if self.source_data_type == np.ndarray:
            data = data.cpu().numpy()
        return data


    def _val_split(self, x_data, y_data, val_split):
        ''' split the data and labels between a training and validation set '''
        assert val_split > 0.0 and val_split < 1.0
        num_records = self._num_records(x_data, y_data)
        val_size = int(num_records * val_split)
        
        # randomly select items for val data
        o = torch.ones(num_records, 
                       dtype = torch.float32,
                       device = self.device()) # equal probs for each item
        t = torch.multinomial(o, val_size) # indexes of sample for validation
        # create an index set for the validation, which allows use of ~ not
        z = torch.zeros(num_records, 
                        dtype = torch.uint8, 
                        device = self.device())
        z[t] = 1 # create the index set
        # split the data with boolean slicing
        #x_val_data = x_data[z]
        x_val_data = self._get_slice(x_data, z)
        y_val_data = y_data[z]
        #x_data = x_data[~z]
        x_data = self._get_slice(x_data, ~z)
        y_data = y_data[~z]
        return x_data, y_data, x_val_data, y_val_data

    def _train_batch(self, x, y):
        ''' Train one batch '''
        self.optimizer.zero_grad()
        x_out = self.model(x)
        loss = self.lossfn(x_out, y)
        loss.backward()
        self.optimizer.step()
        return loss.item(), x_out
    
    def _eval_batch(self, x, y):
        ''' Evaluate one batch '''
        with torch.no_grad():
            x_out = self.model(x)
            loss = self.lossfn(x_out, y)        
        return loss.item(), x_out
    
    def _batch_print_line(self, num_processed, num_records, 
                         loss, tm, verbose, finished = False,
                         metric_values = None,
                         val_loss = None,
                         val_metric_values = None):
        ''' generate the line for showing batch progress '''
        # Quit if no printing needed
        if verbose == 0 or (verbose == 2 and not finished):
            return
        
        # Num batches
        batch_line = str(num_processed) + '/' + str(num_records)
        
        # generate progress line
        line_len = 30
        progress = int(line_len * num_processed / num_records)
        if num_processed == num_records: # Finished [===========================]
            progress_line = ' [' + '=' * line_len + ']'
        else: # in progress  [===============>..............]
            progress_line = ' [' + '=' * progress + '>' + '.' * (line_len - progress - 1) + ']'
            
        # generate time line
        if finished:  # total time plus time per step
            tm_step = int(tm / num_records * 1000000) # in micro seconds
            time_line = ' - ' + str(int(tm)) + 's ' + str(tm_step) + 'us/step'
        else: # show ETA
            eta = int(tm * (num_records - num_processed) / num_processed)
            time_line = ' - ETA: ' + str(eta) +'s'
            
        # generate loss line
        loss_line = ' - loss: {:.4f}'.format(loss)
        
        # Generate metrics
        metrics_line = self._metrics_string(metric_values)
        
        # generate val line
        if val_loss is None:
            val_line = ''
        else:
            val_line = (' - val_loss: {:.4f}'.format(val_loss) + 
                        self._metrics_string(val_metric_values, 'val_'))
            
        
        # print line.  
        # Use stdout rather than print to print to update line inplace
        line = ''
        if verbose == 1: line += batch_line + progress_line
        line += time_line + loss_line + metrics_line + val_line 
        
        stdout.write('\r' + line)
        stdout.flush() # apparently needed on some systems
        stdout.write('\r' + line)
        if finished:
            print('') # for newline
      

    def _validate(self, x_data, y_data, batch_size):
        '''evaluate data, used for validation and evaluation'''
        num_records = self._num_records(x_data, y_data)
        num_batches = int((num_records - 1)/ batch_size) + 1 # last small batch if needed        
        mean_loss = 0
        metric_values = None # for mean metric values
        for batch in range(num_batches):
            # Establish batch start and end
            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size
            if batch_end > num_records: batch_end = num_records
            ixs = slice(batch_start,batch_end)
            loss, x_out = self._eval_batch(self._get_slice(x_data, ixs), 
                                          y_data[ixs])
            metric_values = self._calculate_metrics(x_out, 
                                                    y_data[ixs],
                                                    metric_values,
                                                    batch)
            mean_loss += (loss - mean_loss)/(batch + 1)
        return [mean_loss] + metric_values

    # metrics
    def _accuracy(self, x, y):
        '''Accuracy metric '''
        
        if len(x.size()) == len(y.size()):
            # implies target y one hot encoded
            y_max = y.max(-1)[1]  # argmax, i.e. max category
        elif len(x.size()) - 1 == len(y.size()):
            # implies y categorially encoded
            y_max = y
        else:
            assert False, 'Accuracy metric: Sizes of target and predict incompatible'
            
        x_max = x.max(-1)[1]  #  argmax, i.e. max category
        count_all = np.prod(list(x_max.size()))
        count_correct = (x_max == y_max).sum().item()
        acc = count_correct / count_all
        return acc    

    def _calculate_metrics(self, x, y, metric_values = None, batch = 0):
        ''' return a list of metrics, being the running mean of the 
        previously supplied metric values '''
        mv = []
        # dictionary of metric functions
        metric_dict = {'accuracy':self._accuracy}
        # iterate through metrics
        for i, metric in enumerate(self.metrics):
            if type(metric) == str:
                assert metric in metric_dict, 'Unknown metric: ' + metric
                v = metric_dict[metric](x,y)
            else: # metric should be user defined function
                v = metric(x, y)
            
            # calculate running mean
            if metric_values is not None:
                v = metric_values[i] + (v - metric_values[i]) / batch 

            mv.append(v)
        return mv

    def _metric_name(self, metric):
        abbreviations ={'accuracy':'acc'}
        
        if type(metric) == str:
            name = abbreviations.get(metric, metric)
        else:
            name = metric.__name__
        return name

    def _metric_string(self, value, metric, prefix = ''):
        ''' Create the metrics print string '''
        s = ' - ' + prefix + self._metric_name(metric) + ': {:.4f}'.format(value) 
        return s

    def _metrics_string(self, values, prefix = ''):
        ''' creates the print string for all metrics'''
        s = ''
        for i, metric in enumerate(self.metrics):
            s +=  self._metric_string(values[i], metric, prefix)
        return s
    
    def _num_records(self, x_data, y_data = None, num_records = None):
        ''' Return the number of records from the first axis, confirming that
        all sets have the same size '''
        if type(x_data) in [tuple, list]:
            for x in x_data:
                num_records = self._num_records(x, y_data, num_records) 
        else:
            if num_records is None:
                num_records = x_data.size(0)
                if y_data is not None:
                    assert num_records == y_data.size(0), 'Data and labels must be same size'
            else:
                assert num_records == x_data.size(0), 'All input sets must have same number of records'
        return num_records
    
    def _get_slice(self, x, ixs):
        ''' given data return the indexed elements from the data.
        data may be a tensor of a list of (list of) tensors '''
        if type(x) in [list, tuple]:
            data = [self._get_slice(t, ixs) for t in x]
        else:
            data = x[ixs]
        return data
            
    
    def fit(self, x_data, y_data, 
            validation_split = None, 
            validation_data = None,
            batch_size = 32,
            epochs = 1,
            verbose = 1):
        ''' fit model using training data
        verbose: Integer. 0, 1, or 2. Verbosity mode.
        0 = silent, 1 = progress bar, 2 = one line per epoch.'''
        x_data = self._set_data_type_to_tensor(x_data)
        y_data = self._set_data_type_to_tensor(y_data)
        self.training_mode = self.model.training
        
        # prepare Validation data
        if validation_data is not None:
            assert len(validation_data) == 2
            x_val_data, y_val_data = validation_data   
            x_val_data = self._set_data_type_to_tensor(x_val_data)
            y_val_data = self._set_data_type_to_tensor(y_val_data) 
        elif validation_split is not None:
            x_data, y_data, x_val_data, y_val_data = self._val_split(
                    x_data, y_data, validation_split)
        else:
            x_val_data = None
            y_val_data = None
            val_metric_values = [None]

        # generate history dictionary for results of training
        hist = {}
        hist['loss'] = []
        for metric in self.metrics:
            hist[self._metric_name(metric)] = []
        if x_val_data is not None:
            hist['val_loss'] = []
            for metric in self.metrics:
                hist['val_' + self._metric_name(metric)] = []           

        # training loop
        num_records = self._num_records(x_data, y_data)
        num_batches = int((num_records - 1)/ batch_size) + 1 # last small batch if needed
        ix = list(range(num_records))
        
        for epoch in range(epochs):
            if verbose != 0: print('Epoch {}/{}'.format(epoch + 1, epochs))
            random.shuffle(ix)
            t0 = timer()
            mean_loss = 0
            metric_values = None # reset metric values at epoch start
            self._set_training_mode(True)
            for batch in range(num_batches):
                # Establish batch start and end
                batch_start = batch * batch_size
                batch_end = (batch + 1) * batch_size
                if batch_end > num_records: batch_end = num_records
                
                ixs = ix[batch_start:batch_end]
                loss, x_out = self._train_batch(self._get_slice(x_data, ixs), 
                                                y_data[ixs])
                mean_loss += (loss - mean_loss)/(batch + 1)
                metric_values = self._calculate_metrics(x_out, y_data[ixs],
                                                        metric_values, batch)
                t1 = timer()
                self._batch_print_line(batch_end, 
                                      num_records,
                                      mean_loss,
                                      t1 - t0, verbose, False, 
                                      metric_values = metric_values)
            # update history
            hist['loss'].append(mean_loss)
            for i, metric in enumerate(self.metrics):
                hist[self._metric_name(metric)].append(metric_values[i])
            
            if x_val_data is not None:
                # Validate
                self._set_training_mode(False)
                val_metric_values = self._validate(x_val_data, y_val_data, batch_size)
                # update history
                hist['val_loss'].append(val_metric_values[0])
                for i, metric in enumerate(self.metrics):
                    hist['val_' + self._metric_name(metric)].append(val_metric_values[i + 1])
                    
            t1 = timer()
            self._batch_print_line(num_records, 
                                  num_records,
                                  mean_loss,
                                  t1 - t0, verbose, True,
                                  metric_values = metric_values,
                                  val_loss = val_metric_values[0],
                                  val_metric_values = val_metric_values[1:])
        
        self._reset_training_mode()
            
        return hist
    
    def predict(self, x_data):
        ''' predict output from model '''
        x_data = self._set_data_type_to_tensor(x_data)
        self.training_mode = self.model.training
        self._set_training_mode(False)
        
        with torch.no_grad():
            x = self.model(x_data)
            if self.predictfn is not None:
                x = self.predictfn(x)

        self._reset_training_mode()
        x = self._reset_data_type(x)        
        return x

    def evaluate(self, x_data, y_data, batch_size = 32, verbose = 1):
        ''' evaluate test data with losses and metrics 
        Returns a list of loss plus specified metrics'''
        x_data = self._set_data_type_to_tensor(x_data)
        y_data = self._set_data_type_to_tensor(y_data) 
        self.training_mode = self.model.training
        self._set_training_mode(False)
        num_records = self._num_records(x_data, y_data)
        t0 = timer()
        metric_values = self._validate(x_data, y_data, batch_size)
        t1 = timer()
        self._batch_print_line(num_records, 
                              num_records,
                              metric_values[0],
                              t1 - t0, verbose, True,
                              metric_values = metric_values[1:])
        
        self._reset_training_mode()
        return metric_values  # loss plus metric values
        
    def save(self, file):
        ''' Save the model and optimizer parameters etc'''
        d = [self.model.state_dict(), self.optimizer.state_dict()]
        with open(file, 'wb') as f:
            torch.save(d, f)
    
    def load(self, file):
        ''' Load the model and optimizer parameters from file '''
        with open(file, 'rb') as f:
            if torch.cuda.is_available():
                d = torch.load(f)
            else:
                d = torch.load(f, map_location = 'cpu')        
        self.model.load_state_dict(d[0])
        self.optimizer.load_state_dict(d[1])


#%% Unit tests
    
class BasicModel(nn.Module):
    ''' Basic Model for testing'''
    def __init__(self):
        super(BasicModel, self).__init__()
        self.fc1 = nn.Linear(5, 2)
        
    def forward(self, inputs):
        return self.fc1(inputs)


def tensor_equal(t1, t2):
    ''' return True if 2 tensors are always equal '''
    return (t1 == t2).min().item()
    
class TestValSplit(unittest.TestCase):
    def test_val_split(self):
        ''' test validation split, by splitting and then reassembling splits
        and comparing to original '''
        model = BasicModel()
        cmodel = CompiledModel(model, None, None, [])
        x_data = torch.tensor([[6.0, 6.1], [7.0, 7.1], [8.0, 8.1], 
                               [9.0, 9.1], [10.0, 10.1]])
        y_data = torch.tensor([6, 7, 8, 9, 10])
        validation_split = 0.45  # will round down to 0.4, i.e. size 2
        x, y, x_val, y_val = cmodel._val_split(x_data, y_data, validation_split)
        assert x.size(0) == 3
        assert y.size(0) == 3
        assert x_val.size(0) == 2
        assert y_val.size(0) == 2
        
        # reconstruct original data from split and check they match
        x_all = torch.cat((x, x_val), dim = 0) 
        x_all = x_all.sort(dim = 0)[0] # To realign with x_data which is already sorted
        assert tensor_equal(x_data, x_all)
        y_all = torch.cat((y, y_val), dim = 0) 
        y_all = y_all.sort(dim = 0)[0] # To realign with x_data which is already sorted
        assert tensor_equal(y_data, y_all)
        

class TestAccuracyMetric(unittest.TestCase):
    def test_accuracy_one_hot_encoded(self):
        ''' Test accuracy where target is one hot encoded '''
        model = BasicModel()
        cmodel = CompiledModel(model, None, None, ['accuracy'])
        x_data = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], 
                               [1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])        
        y_data = torch.tensor([[1, 0, 0], [1, 0, 0], 
                               [0, 0, 1], [0, 1, 0]])
        accuracy = cmodel._calculate_metrics(x_data, y_data)[0] 
        assert accuracy == 0.75
        
        
    def test_accuracy_sparse_encoded(self):
        '''Test accuracy where target is sparsely encoded '''
        model = BasicModel()
        cmodel = CompiledModel(model, None, None, ['accuracy'])
        x_data = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], 
                               [1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])        
        y_data = torch.tensor([0, 0, 2, 1])
        accuracy = cmodel._calculate_metrics(x_data, y_data)[0] 
        assert accuracy == 0.75        

class TestNumRecords(unittest.TestCase):
    def test_num_records(self):
        model = BasicModel()
        cmodel = CompiledModel(model, None, None, [])
        x_data1 = torch.tensor([[6.0, 6.1], [7.0, 7.1], [8.0, 8.1], 
                               [9.0, 9.1], [10.0, 10.1]])
        x_data2 = torch.tensor([[1.0, 1.1], [7.0, 7.1], [8.0, 8.1], 
                               [9.0, 9.1], [1.0, 1.1]])
        x_data = [x_data1, (x_data1, x_data2)]
        y_data = torch.tensor([6, 7, 8, 9, 10])
        num_records = cmodel._num_records(x_data, y_data)
        assert num_records == 5

    def test_set_data_type_to_tensor(self):
        ''' Test conversion of arrays and lists of arrays to tensors '''
        model = BasicModel()
        cmodel = CompiledModel(model, None, None, [])
        x_data1 = np.array([[6.0, 6.1], [7.0, 7.1], [8.0, 8.1], 
                               [9.0, 9.1], [10.0, 10.1]], dtype = np.float32)
        x_data2 = np.array([[1.0, 1.1], [7.0, 7.1], [8.0, 8.1], 
                               [9.0, 9.1], [1.0, 1.1]], dtype = np.float32)
        x_data = [x_data1, (x_data1, x_data2)]
        y_data = np.array([6, 7, 8, 9, 10], dtype = np.int64)
        x_data_T = cmodel._set_data_type_to_tensor(x_data)
        y_data_T = cmodel._set_data_type_to_tensor(y_data)
        num_records = cmodel._num_records(x_data_T, y_data_T)
        assert num_records == 5
        assert type(y_data_T) == torch.Tensor
        assert type(x_data_T[0]) == torch.Tensor
    
    def test_get_slice(self):
        model = BasicModel()
        cmodel = CompiledModel(model, None, None, [])
        x_data1 = torch.tensor([[6.0, 6.1], [7.0, 7.1], [8.0, 8.1], 
                               [9.0, 9.1], [10.0, 10.1]])
        x_data2 = torch.tensor([[1.0, 1.1], [7.0, 7.1], [8.0, 8.1], 
                               [9.0, 9.1], [1.0, 1.1]])
        x_data = [x_data1, (x_data1, x_data2)]
        ixs = [0, 4]
        x_batch1 = torch.tensor([[6.0, 6.1], [10.0, 10.1]])        
        batch = cmodel._get_slice(x_data, ixs)
        num_records = cmodel._num_records(batch)
        assert num_records == 2
        assert tensor_equal(x_batch1, batch[0])


        
if __name__ == '__main__':
    unittest.main()
         




#%% testing
