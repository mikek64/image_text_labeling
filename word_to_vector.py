# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:31:24 2018

@author: mike_k

encodes words as vectors
"""

import numpy as np
import re
from dateutil.parser import parse
import config

#%% Get weights embedding as dictionary

word2vec = {}
with open(config.local_cfg['WORD_VEC'], encoding='utf8') as f:
    for line in f.readlines():
        values = line.split()
        word = values[0]
        vec = np.array(values[1:], dtype='float32')
        word2vec[word] = vec


if __name__ == '__main__':
    print(len(word2vec))

    # Test word vectorisation
    assert 'invoice' in word2vec
    assert not 'Invoice' in word2vec  # words must in lower case
    assert not 'invoice ' in word2vec  # trim first
    assert not 'invoice:' in word2vec  # remove parts of speech
    assert '1' in word2vec # some numbers in word to vec
    assert '9' in word2vec
    assert '255' in word2vec # largeish numbers in word to vec
    assert '1024' in word2vec
    assert not '6523' in word2vec
    assert '3.59' in word2vec  # decimals
    assert not '255.43' in word2vec # not decimals
    assert ',' in word2vec
    assert '#' in word2vec
    assert '.' in word2vec
    assert ':' in word2vec
    assert ',' in word2vec
    assert '$' in word2vec
    assert '£' in word2vec
    
#%%
# Embedding defn
# derive size from GloVe
# determine embedding size from random element from embedding
EMBEDDING_DIM = len(next(iter(word2vec.values()))) # size of GloVe embedding(50)
EMBED_ISNUMERIC = EMBEDDING_DIM + 0  # 1 as isnumeric
EMBED_ISDOT = EMBEDDING_DIM + 1  # 1 as dot or comma
EMBED_ISSEPARATOR = EMBEDDING_DIM + 2  # 1 as separator colon, semicolon
EMBED_ISSYMBOL = EMBEDDING_DIM + 3  # 1 as currency symbol
EMBED_ISWORD = EMBEDDING_DIM + 4  # 1 if its a word (to distinguish from empty space)
EMBED_ISDATE = EMBEDDING_DIM + 5  # 1 if its a date
# Total embedding length.  Also defines the INPUT_SIZE in ml for the model
EMBED_SIZE = EMBEDDING_DIM + 6  # Total embedding length. 


#%%  Create numpy array for word embedding

# word embedding is numpy array num words by EMBED_SIZE

def is_date(s):
    ''' returns True if date is a string '''
    try:
        dt = parse(s)
        return True
    except:
        return False

def embed_words(words):
    ''' embed words using specific flags and GloVE.  If date or number flag 
    is set then  all other flags and GloVe do not get set'''
    n = len(words)
    wordsembed = np.zeros((n, EMBED_SIZE), dtype = np.float32)
    words_lower = [w.lower().strip() for w in words] # convert to lower case and strip ending spaces
    
    # Encode each word
    for i in range(n):
        # flag for whether encoding has already been set
        if wordsembed[i,EMBED_ISDATE] == 1:  # can only have isdate set in advance
            is_set = True
        else:
            is_set = False
        
        # get current and next 2 words
        w = words_lower[i]
        # next word
        if i<n-1:
            w2 = words_lower[i+1]
        else:
            w2 = ''
        # subsequent word
        if i<n-2:
            w3 = words_lower[i+2]
        else:
            w3 = ''        
        
        # test for date first, testing one 2 and 3 consecutive characters
        if is_date(w):
            wordsembed[i,EMBED_ISDATE] = 1
            is_set = True
        elif w2 != '' and is_date(w + ' ' + w2):
            wordsembed[i:i+2,EMBED_ISDATE] = 1
            is_set = True
        elif w3 != '' and is_date(w + ' ' + w2 + ' ' + w3):
            wordsembed[i:i+3,EMBED_ISDATE] = 1
            is_set = True            
        
        
        # test for number, stripping numbers and separators pre test
        wn = ''.join([x for x in w if x not in '£$,.'])
        if wn.isnumeric() and not is_set:
            wordsembed[i,EMBED_ISNUMERIC] = 1
            is_set = True

        if len(w) == 1 and not is_set:
            # test for specific characters to encode.  These can be
            # GloVe encoded.  no neede to use is_set 
            if w in ',.':
                wordsembed[i,EMBED_ISDOT] = 1
            elif w in ':;_-':
                wordsembed[i,EMBED_ISSEPARATOR] = 1
            elif w in '£$':
                wordsembed[i,EMBED_ISDOT] = 1
        wordsembed[i,EMBED_ISWORD] = 1 # always encode as a word
        if w in word2vec and not is_set: # use word vectorisation if not a number/date
            wordsembed[i,:50] = word2vec[w]
        else:
            w1 = re.sub('\W','',w) 
            if w1 in word2vec and not is_set: # strip non alpha characters and try again
                wordsembed[i,:50] = word2vec[w1]

    return wordsembed

#%%  Test
if __name__ == '__main__':
    from read_wordtable import read_wordtable_to_words_and_locations as read_table
    words, _ = read_table(config.local_cfg['TEST_FILE'], config.pathtable)

    wordsembed = embed_words(words)
    assert wordsembed.shape == (len(words),EMBED_SIZE)
