# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:04:05 2018

@author: mike_k
"""
import config
import numpy as np
#%%
if __name__ == '__main__':
    TESTFILE = 'inv_5_2'

#%%
#  Read word table 
    
def read_wordtable_to_words_and_locations(file, path):
    with open(path + '/' + file + '_table.txt','r', encoding = 'utf-8') as f:
        t = f.read()
    lines = t.split('\n')
    n = len(lines)
    words = []
    location = np.zeros((n-1,8))
    for i in range(1,n):
        line = lines[i].split('\t')
        words.append(line[0])
        for j in range(8):
            location[i-1,j] = line[j+1]
    return words, location
    
if __name__ == '__main__': 
    words, location = read_wordtable_to_words_and_locations(TESTFILE, 
                                                            config.pathtable)
    print(words[0:5], location[0:5])