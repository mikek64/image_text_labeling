# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:28:04 2019

@author: mike_

Verifies the training tags are the word vector file
Missing tags will cause unassigned words to be potentially identified as tags
Any such missing words should be added to the vectorization file
"""

import config
import os

from generate_training_data import tags_folder, TAGS_LIST


#%%


class Tags():
    ''' Verifies the training tags are the word vector file'''
    def __init__(self):
        
        # load tags
        with open(os.path.join(tags_folder, TAGS_LIST)) as f:
            self.tags = [x.split(',') for x in f.read().split('\n')[1:]]  # discard headers
        
        self.tags_lists = [(self.load_tag_file(t[0])) for t in self.tags]
        

        # load word vectors
        words = []
        with open(config.local_cfg['WORD_VEC'], encoding='utf8') as f:
            for line in f.readlines():
                values = line.split()
                words.append(values[0].lower())
                

        self.words = set(words)
    
        self.check_all_tag_files()
        
    
    def load_tag_file(self, tag_file):
        ''' open tag file and load as list of list of words'''
        wds = []
        with open(os.path.join(tags_folder, tag_file)) as f:
            for x in f.read().split('\n'):
                wds += x.strip().split(' ')
        return wds

    def check_all_tag_files(self):
        ''' for all tag files check words are in vector file '''
        self.inlist = []
        self.outlist = []
        for tags in self.tags_lists:
            intag, outtag = self.check_tag_file(tags)
            self.inlist.append(intag)
            self.outlist.append(outtag)
        

    def check_tag_file(self, tags):
        ''' check words in tags are in vector '''
        intag = []
        outtag = []
        for w in tags:
            if w.lower() in self.words:
                intag.append(w)
            else:
                outtag.append(w)
        return intag, outtag
      
    def print_results(self):
        exceptions = False
        for i, t in enumerate(self.tags):
            if len(self.outlist[i]) !=0:
                exceptions = True
                print(t[0] + ": " + ", ".join(self.outlist[i]))
        
        if not exceptions:
            print('All tags in word vectors')
            

if __name__ == '__main__':
    tgs = Tags()
    tgs.print_results()
   