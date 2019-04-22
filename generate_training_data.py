# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:32:53 2019
This file generates training data for the modeller

@author: mike_k
"""

import numpy as np
import os
from sys import stdout
import config
import random
import datetime
from json_to_table import save_word_table
import table_to_array
#%%
TAGS_LIST = 'tags.csv' 

# Contents of the tags list
T_FILE = 0
T_PROB = 1
T_VAL_LBL = 2
T_LABEL_LBL = 3
T_TYPE = 4
T_PARAM_1 = 5
T_PARAM_2 = 6


# Max spacing
WORD_SPACING = 3 # number of characters
TAG_VALUE_SPACING = 30 # max number of characters between label and the value
BLOCK_SPACING = 10    # minimum interblock spacing
MIN_ROW_WIDTH = 100 # minimum width of rows
MAX_BLOCK_WORDS = 8 # max number of words in a block

#%%
path_table = config.pathtable
path_label = config.pathlabel
tags_folder = config.local_cfg['TRAINING_DATA_GENERATION']

        
#%%

class Values():
    ''' generate values for sample generation '''
    def __init__(self):
        self.dateformats = ['%d %B %Y', # dd mmmm yyyy
                           '%d %m %Y', # dd mm yyyy 
                           '%d %b %Y', # dd mmb yyyy 
                           '%d/%m/%Y', # dd/mm/yyyy
                           '%d %B %y', # dd mmmm yy
                           '%d %b %y', # dd mmm yy
                           '%d %m %y', # dd mm yy
                           '%d/%m/%y'] # dd/mm/yy
        
        self.currencies = ['£','$','']

    def sample(self, v_type = None, param1 = None, param2 = None):
        
        if v_type is not None:
            v_type = v_type.strip()
        r = random.random() # Random sample
        if v_type == 'decimal' or (v_type is None and r < 0.5):
            v = self.sample_decimal(param1, param2)
        elif v_type == 'mixed'or (v_type is None and r < 0.8):
            v = self.sample_mixed(param1, param2)
        elif v_type == 'date'or v_type is None:
            v = self.sample_date()         
        else:
            assert False, 'Unknown tag value type'   
        return v.split(' ')

    def sample_date(self): 
        ''' create sample date '''
        d = datetime.date.today() 
        d += datetime.timedelta(days=random.randint(-2000,2000))
        return  d.strftime(random.sample(self.dateformats, 1)[0]) 
    
    def sample_decimal(self, param1, param2):
        ''' create decimal up to param1 digits and param2 decimals '''
        if param1 is None:
            param1 = '7'  # up to 7 figures
        if param2 is None:
            param2 = '2'  # 2 decimal places

        num_max = 10 ** random.randint(1,int(param1))
        num = random.randint(1, num_max - 1)
        if param2 != 0:  # add decimals
            num_dec = float(random.randint(0, 10 ** int(param2) - 1)) / 10 ** int(param2)
            num += num_dec
        
        if int(param2) == 0:
            fmt = '{:,}'
        else:
            fmt = '{:,.' + param2.strip() + 'f}'
        
        currency = random.sample(self.currencies, 1)[0]
        
        return currency + fmt.format(round(num,2))
    
    def sample_mixed(self, param1, param2):
        ''' created mixed characters of format ABC12345 '''
        if param1 is None:
            param1 = '2'  #up to 2 letters
        if param2 is None:
            param2 = '7' #Up to 7 numbers 
        
        s = ''
        for i in range(0, random.randint(0, int(param1))):
            s += chr(random.randint(65,90))  # Capital letters
        for i in range(0, random.randint(1, int(param2))):
            s += str(random.randint(0,9))  # Numbers
   
        return s

if __name__ == '__main__':
    vls = Values()
    print(vls.sample('date', '7', '2'))
    print(vls.sample())

#%%

class Words():
    ''' list of words with random word sample generation 
    sample includes groups and labels'''
    def __init__(self):
        with open(config.local_cfg['WORDS']) as f:
            self.words_list = f.read().split('\n')
    
    def sample(self, n):
        ''' random sample of up to n words '''
        r = random.randint(1, n)
        wd = random.sample(self.words_list, r)
        lb = ['1' for x in wd]  # label as 1, unassigned
        gs = [0 for x in wd]
        return wd, lb, gs
        
    
    def sample_tags(self):
        ''' sample of words tagged to a value with groups '''
        r = random.randint(1, 3)
        wd = random.sample(self.words_list, r)
        vs = vls.sample()
        gs = [0 for x in wd] + [1 for x in vs]
        lb = ['1' for x in gs]  # label as 1, unassigned
        
        return wd + vs, lb, gs
        
if __name__ == '__main__':        
    wds = Words()
    print(wds.sample(10))

#%%

class Tags():
    '''  list of tags with function that can create a sample of
    data and its tag by randomly sampling from the tag lists'''
    def __init__(self):
        
        # self.tags are the tag files as
            # File Name
            # Probability of being used
        with open(os.path.join(tags_folder, TAGS_LIST)) as f:
            self.tags = [x.split(',') for x in f.read().split('\n')[1:]]  # discard headers
        
        self.tags_lists = [(self.load_tag_file(t[0])) for t in self.tags]

    
    def load_tag_file(self, tag_file):
        ''' open tag file and load as list of list of words'''
        with open(os.path.join(tags_folder, tag_file)) as f:
            return [x.split(' ') for x in f.read().split('\n')]

    def sample(self):
        ''' return a sample of items from the tags '''
        s = []
        label_s = []
        group_s = []  # A grouping to distinguish labels from values
        
        for i, t in enumerate(self.tags):
            if random.random() < float(t[T_PROB]):
                # sample words
                ws = self.tags_lists[i][random.randint(0, 
                                   len(self.tags_lists[i]) - 1)]
                # sample value
                vs = vls.sample(t[T_TYPE], t[T_PARAM_1],t[T_PARAM_2])

                smpl = ws + vs
                
                label_w = [t[T_LABEL_LBL] for x in ws]
                label_v = [t[T_VAL_LBL] for x in vs]
                labels = label_w + label_v
                
                group_w = [0 for x in ws]
                group_v = [1 for x in vs]
                groups = group_w + group_v
                
                s.append(smpl)
                label_s.append(labels)
                group_s.append(groups)
        return s, label_s, group_s
      

if __name__ == '__main__':
    tgs = Tags()
    print(tgs.sample())

#%%

class Document():
    ''' A document to be generated '''
    def __init__(self):
        self.n_rows = random.randint(20, 50) # number of rows in the invoice
        self.row_gap = 10.0
        self.row_height = 9.0
        
        # letter separators
        self.letter_gap = random.randint(1, WORD_SPACING)
        
        # sample of tags with labels
        self.ws, self.vs, self.gs = tgs.sample() # sample of words and labels
        self.get_words_and_labels()
        # list of rows, and the tag index in which each sample appears
        self.tag_rows, self.tag_indx = self.assign_sample_to_rows() 
        self.get_words_and_rows()
        self.get_min_row_length()
        
        # set row width ensuring it is wide enough to include all rows
        self.row_length = max(MIN_ROW_WIDTH, 
                              self.overall_min_row_length + 
                              BLOCK_SPACING + 
                              random.randint(0, 50))
        # turn the words into a words list and location array
        self.create_word_list()

    
    def get_words_and_labels(self):
        ''' get all the blocks of text and tags/labels.  Note
        this is based on having  up to 2 blocks in each row '''
        
        # sample of word blocks
        m = random.randint(5, self.n_rows)

        for i in range(m):
            word_blocks, word_labels, word_groups = wds.sample(MAX_BLOCK_WORDS)
            self.ws += [word_blocks]
            self.vs += [word_labels]
            self.gs += [word_groups]
        
        #sample of tag blocks
        m = random.randint(2, int(self.n_rows/3))
        for i in range(m):
            tag_blocks, tag_labels, tag_groups = wds.sample_tags()
            self.ws += [tag_blocks]
            self.vs += [tag_labels]
            self.gs += [tag_groups]
        
    def assign_sample_to_rows(self):
        ''' returns numbers representing the row in which each word block
        appears.  Can have up to 2 word blocks on each row '''
        rownums = list(range(self.n_rows * 2)) # numbers 0 to n_rows twice
        r = random.sample(rownums, len(self.ws))
        random.shuffle(r)
        
        # resort r and its index ascending
        ix = sorted(range(len(r)), key=lambda k: r[k])
        r.sort()
        return r, ix

    def get_words_and_rows(self):
        ''' take the blocks and make a word and row list '''
        # word, row, word length, block index, is value
        self.word_table = []
        # for each row: total word length, num words, 
        # num blocks, num value blocks,
        # Tag spacing, total min length
        self.rows = [[0, 0, 0, 0, 
                      random.randint(WORD_SPACING, TAG_VALUE_SPACING), 
                      0] for x in range(self.n_rows)] # row length
        # the labels
        self.labels = []
        
        for i, rv in enumerate(self.tag_rows):
            rw = int(rv/2) # up to two blocks per row
            ix = self.tag_indx[i]
            self.rows[rw][2] += 1  # block count
            if self.gs[ix][-1] == 1: # includes a value
                self.rows[rw][3] += 1 # label count
            for j, wd in enumerate(self.ws[ix]):
                self.word_table.append([wd, rw, len(wd), i, self.gs[ix][j]])
                self.rows[rw][0] += len(wd)  # total word length for row
                self.rows[rw][1] += 1  # word count for row
                
                self.labels.append(int(self.vs[ix][j].strip()))
                
    def get_min_row_length(self):
        ''' for each row update the min row length '''
        self.overall_min_row_length = 0
        for i, r in enumerate(self.rows):
            r[5] = (r[0] + # word length
                         max((r[1] - 1 - r[3]),0) * self.letter_gap + # spaces
                         r[3] * r[4]    # label value spacing (wider than normal)
                         )
            if r[5] > self.overall_min_row_length:
                self.overall_min_row_length = r[5]
    
    def create_word_list(self):
        ''' generate word list '''
        
        self.words = []
        self.location = np.zeros((len(self.word_table),8))
        
        # header
        s = 'text\tx-top-left\ty-top-left\tx-top-right\ty-top-right'
        s += '\tx-bottom-right\ty-bottom-right\tx-bottom-left\ty-bottom-left'
        cur_row = -1
        cur_block = 0
        cur_group = 0
        bottom = 0.
        top = 0.
        left = 0.
        right = 0.
        
        for i, w in enumerate(self.word_table):
            self.words.append(w[0])
            #s += '\n' + w[0] 
            rw = self.rows[w[1]]
            if w[1] != cur_row: # New row
                cur_row = w[1]
                cur_block = w[3]
                cur_group = 0
                # establish top bottom & left
                top = cur_row * self.row_gap
                bottom = top + self.row_height
                left = 0.
                # establish gap and left position
                gap = self.row_length - rw[5]
                n_blocks = rw[2]
                if n_blocks == 2:
                    mid_gap = random.randint(BLOCK_SPACING, gap)
                    gap = gap - mid_gap
                left = random.randint(0, gap)
            else:
                if w[3] != cur_block: # new block
                    cur_block = w[3]
                    cur_group = 0
                    left = right + mid_gap
                elif w[4] != cur_group:  # new group, hence a value item
                    cur_group = w[4]
                    left = right + rw[4] # add specific tag spacing
                else:
                    left = right + self.letter_gap
                    
            right = left + w[2]
            
            # add numbers
            self.location[i,0] = left
            self.location[i,1] = top
            self.location[i,2] = right
            self.location[i,3] = top
            self.location[i,4] = right
            self.location[i,5] = bottom
            self.location[i,6] = left
            self.location[i,7] = bottom 

            nf = '{:.1f}'
            s += '\t' + nf.format(left)
            s += '\t' + nf.format(top)
            s += '\t' + nf.format(right)
            s += '\t' + nf.format(top)
            s += '\t' + nf.format(right)
            s += '\t' + nf.format(bottom)
            s += '\t' + nf.format(left)
            s += '\t' + nf.format(bottom)   
    
    def save_tables(self, file):
        ''' save the tables to the file '''
        save_word_table(file, path_table, self.words, self.location)
        self.save_labels(file)
        
    def save_labels(self,file):
        s = 'Labels'
        d = self.label_descriptions() # disctionary of label to a description
        for lb in self.labels:
            s += '\n' + d[lb]
        
        # Save as csv
        with open(path_label + '/' + file + '_labels.csv','w', encoding = 'utf-8') as f:
            f.write(s)
            
    def label_descriptions(self):
        
        # read label master file ignoring headers
        with open(config.local_cfg['LABEL_MASTER'],'r', encoding = 'utf-8') as f:
            t = f.read().split('\n')[1:]
        # load to a dictionary of code, description
        d = {}
        for x in t:
            k,v = x.split(',')
            d[int(k)] = v
        return d
         
        


         
if __name__ == '__main__':              
    d = Document()  
    print(d.words)
    print(d.location)
    # d.save_tables('test_01')


#%%

# Generate test sample

DATA_SIZE = 10000
START = 1
BATCH = '01'

if __name__ == '__main__': 
    for i in range(DATA_SIZE):
        file = 'generated_' + BATCH + '_' + ('00000' + str(i + START))[-5:]
        d = Document()
        d.save_tables(file)
        stdout.write('\r' + str(i+1) + '/' + str(DATA_SIZE)) # show progress
        stdout.write('')

#%%  Create arrays

if __name__ == '__main__':
    table_to_array.create_all_arrays(config.pathtable,
                                     config.pathlabel,
                                     config.pathwordmap,
                                     config.patharray,
                                     config.pathlabelarray)    
    
