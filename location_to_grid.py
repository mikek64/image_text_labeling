# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:49:58 2018

@author: mike_k

Maps locations to grid
"""
import numpy as np
import config
from matplotlib import pyplot as plt
import math
# for testing only   
from read_wordtable import read_wordtable_to_words_and_locations as read_table

#%%

GRID_SIZE = 64  # the size of the array used for storing the invoice
# vertex order in location file as defined from Google Vision
TOPLEFTX = 0
TOPLEFTY = 1
TOPRIGHTX = 2
TOPRIGHTY = 3
BOTTOMRIGHTX = 4
BOTTOMRIGHTY = 5
BOTTOMLEFTX = 6
BOTTOMLEFTY = 7
# vertex order in position file
TOP = 0
LEFT = 1
BOTTOM = 2
RIGHT = 3


#%%

def mapping_completeness(mpa, location):
    ''' given a mapping array mpa, determines whether all words are included 
    somewhere in the mapping '''
    unmapped = []
    for i in range(location.shape[0]):
        if not i in mpa:
            unmapped.append(i)
    complete = len(unmapped) == 0
    return complete, unmapped

def show_array(map_ay):
    ''' plot array on screen '''
    array_plot = (map_ay != -1) * 128 # words as light, no word as black
    plt.figure(figsize = (8,8))
    plt.imshow(array_plot)    

#%% Version 5 mapping,  align items before snapping to grid
# I.e. enforce items to be on same line that overlap
# Map to whole line by expanding left and right from center    

def are_aligned(item_a, item_b):
    ''' given 2 items: item[0] = top, item[1] =  bottom location, 
    shows whether these overlap 
    sufficiently to be the same word row. Returns True/False'''
    height_a = item_a[1] - item_a[0] # Bottom - top
    height_b = item_b[1] - item_b[0] # Bottom - top
    overlap = min(item_a[1] - item_b[0], item_b[1] - item_a[0])
    min_height = min(height_a, height_b)
    max_height = max(height_a, height_b)
    
    aligned = (min_height / max_height > 0.8 and  # Must be similar size
               overlap / min_height > 0.8) # Must mostly overlap
    
    return aligned

def match_an_aligned(item, not_aligned):
    ''' check whether an item is aligned to any previous item, in not_aligned '''
    assert type(not_aligned) == list
    for e in reversed(not_aligned): # reversed as likely to match recent
        if are_aligned(item, e):
            return True, e
    return False, item
    

def get_map(location, multi_pixel = True):
    ''' map to grid.  Ensure items which line up appear
    on same row.  Handle duplicates '''
    n = location.shape[0]
    # default output -1 means no word present at that pixel
    map_ay = np.ones((GRID_SIZE,GRID_SIZE), dtype = np.int64) * -1

    # indexes in location_detail
    TOPL = 0
    BOTTOML = 1
    WIDTH = 2
    HEIGHT = 3
    LEFTPIXEL = 4
    RIGHTPIXEL = 5
    
    # Get scaling factors
    location_x = location[:,[TOPLEFTX,
                             TOPRIGHTX,
                             BOTTOMRIGHTX,
                             BOTTOMLEFTX]]
    
    location_y = location[:,[TOPLEFTY,
                             TOPRIGHTY,
                             BOTTOMRIGHTY,
                             BOTTOMLEFTY]]
    
    mxh = np.max(location_y) # max height
    mxw = np.max(location_x) # max width
    mnh = np.min(location_y) # min height
    mnw = np.min(location_x) # min width

    # Grid of TOP, BOTTOM location, width middle pixel, height middle pixel
    # left pixel, right pixel
    items = np.zeros((n,6), dtype = int)
    items[:,TOPL] = (location[:,TOPLEFTY] +
                              location[:,TOPRIGHTY])/2
    items[:,BOTTOML] = (location[:,BOTTOMLEFTY] +
                                 location[:,BOTTOMRIGHTY])/2
    # Middle pixel from width
    items[:, WIDTH] = np.round((np.sum(location_x,1)/4 - mnw) * 
                                (GRID_SIZE-2)/(mxw - mnw), 0).astype(int)
    # Middle pixel height
    items[:, HEIGHT] = np.round((np.sum(location_y,1)/4 - mnh)* 
                                (GRID_SIZE-1)/(mxh - mnh) ,0).astype(int)
                   
    # Left pixel
    items[:, LEFTPIXEL] = np.round(((location[:,TOPLEFTX] +
                                              location[:,BOTTOMLEFTX])/2
                                              - mnw) * 
                                (GRID_SIZE-2)/(mxw - mnw), 0).astype(int)
    items[:, RIGHTPIXEL] = np.round(((location[:,TOPRIGHTX] +
                                              location[:,BOTTOMRIGHTX])/2
                                              - mnw) * 
                                (GRID_SIZE-2)/(mxw - mnw), 0).astype(int)    

    
    not_aligned = [] # list of elements, not aligned
    
    # Plot middle pixel first, aligning height to previously plotted items
    # to resolve rounding issues where rounding would place similar items on 
    # different rows.  
    # Also ensure no duplicate locations by shifting duplicates to right
    for i in range(n):
        is_match, matched_item = match_an_aligned(items[i,:], not_aligned)
        if is_match: # Item matches existing one, use that one, instead of own
            items[i,HEIGHT] = matched_item[HEIGHT] # Use matched position height
        else:
            not_aligned.append(items[i,:]) # record item as new line
        # check for duplicate locations and resolve by moving affecting word 1 to the right
        while map_ay[items[i,HEIGHT]][items[i,WIDTH]] != -1: # already mapped that pixel
            items[i,WIDTH] += 1  # next pixel to right
        map_ay[items[i,HEIGHT]][items[i,WIDTH]] = i  # add to map array

    # extend from middle pixels to left and right to draw line
    # do not overwrite existing pixels
    if multi_pixel:
        for i in range(n):
            # extend from center left until hit end of mapped pixel
            for j in range(items[i, WIDTH] - 1,items[i,LEFTPIXEL] - 1, -1):
                if map_ay[items[i,HEIGHT]][j] == -1: # not mapped that pixel
                    map_ay[items[i,HEIGHT]][j] = i
                else:
                    break
            # extend from center right until hit end of mapped pixel
            for j in range(items[i, WIDTH] + 1,items[i,RIGHTPIXEL] + 1, 1):
                if map_ay[items[i,HEIGHT]][j] == -1: # not mapped that pixel
                    map_ay[items[i,HEIGHT]][j] = i
                else:
                    break

    return map_ay

if __name__ == '__main__':
    # Save map as csv to examine later
    _, location = read_table(config.local_cfg['TEST_FILE'], config.pathtable)
    array = get_map(location, True)    
    np.savetxt(config.pathtxtarray + '/' 
               + config.local_cfg['TEST_FILE'] + '_wordmap_np.csv',
               array, delimiter = ',') 
    complete, _ = mapping_completeness(array, location)
    show_array(array) # plot on screen
    print('Mapping completeness:', complete)
    

#%%  ##################### Superceded old versions / test versions ############################

#%% Rotate table

def rotate_point(base_point, round_point, angle):
    ''' rotate the base point (x,y) round the round point (x0, y0) by the angle
    '''
    x, y = base_point
    x0, y0 = round_point

    z = math.sqrt((x-x0)**2 + (y-y0)**2)
    if x == x0:
        a = 3.14159 / 2
    else:
        a = math.atan((y-y0)/(x-x0))

    a += angle
    x1 = x0 + z * math.cos(a)
    y1 = y0 + z * math.sin(a)
    return x1, y1

def nearly_equal(a,b):
    return (abs(a[0] - b[0]) + abs(a[1] - b[1])) < 0.0001 * max(a[0],a[1], b[0], b[1])

if __name__ == '__main__':
    # testing
    assert nearly_equal((2.,3.),(1.99999,3.00001))
    assert not nearly_equal((2.,3.),(2.5,3.1))

    assert nearly_equal((-1.,2.), rotate_point((2.0,1.0), (0.0,0.0), 3.14159/2))
    assert nearly_equal((3.,2.), rotate_point((3.0,2.0), (1.0,1.0), 0))  
    assert nearly_equal((0.,3.), rotate_point((3.0,2.0), (1.0,1.0), 3.14159/2))  

def rotate(location):
    ''' Rotate the location table so image is horizontally aligned, in order
    that words on the same row can be properly aligned.
    Strategy:
        Ascertain overall image rotation as most common rotation from each
        words rotation (after rounding to nearest degree)
        Rotate entire image by this amount.  I.e. rotate all locations round
        0,0 by this amount
        Then rotate each word to be horizontal by rotating the word round its
        mid point'''
    # Programming O/S
        
    return location


#%% Version 4 mapping, map whole image
# New test version of code
# Still have an issue here on how to handle location overlap
def get_map4(location):
    ''' maps to rectangular block. mapping all pixels based on word
    with max % at each pixel.  Assumes image is orientated perpendicular '''
    n = location.shape[0]
    grid = np.ones((GRID_SIZE,GRID_SIZE), dtype = np.int64) * -1
    # initially map to 10 * larger grid and then sum elements mapped
    # to work out % mapped to each pixel
    large_grid = GRID_SIZE * 10
    mapper = np.zeros((large_grid,large_grid, n), dtype = np.int64)
    
    position = np.zeros((n,4), dtype = int) # Grid of TOP, LEFT, BOTTOM, RIGHT
    
    # Get scaling factor
    location_x = location[:,[TOPLEFTX,
                             TOPRIGHTX,
                             BOTTOMRIGHTX,
                             BOTTOMLEFTX]]
    
    location_y = location[:,[TOPLEFTY,
                             TOPRIGHTY,
                             BOTTOMRIGHTY,
                             BOTTOMLEFTY]]
    
    mxh = np.max(location_y) # max height
    mxw = np.max(location_x) # max width
    mnh = np.min(location_y) # min height
    mnw = np.min(location_x) # min width

    # Get top, bottom, left and right positions on the large_grid
    position[:,TOP] = np.round(((location[:,TOPLEFTY] +
                                 location[:,TOPRIGHTY])/2 -
                                mnh) * 
                                (large_grid - 1)/(mxh - mnh),0).astype(int)
    position[:,BOTTOM] = np.round(((location[:,BOTTOMLEFTY] + 
                                    location[:,BOTTOMRIGHTY])/2 -
                                    mnh) * 
                                (large_grid - 1)/(mxh - mnh) + 
                                0.5, 0).astype(int)
    position[:,LEFT] = np.round(((location[:,TOPLEFTX] +
                                  location[:,BOTTOMLEFTX])/2 - 
                                  mnw) * 
                                  (large_grid - 1)/(mxw - mnw), 0).astype(int)
    position[:,RIGHT] = np.round(((location[:,TOPRIGHTX] +
                                   location[:,BOTTOMRIGHTX])/2 - 
                                   mnw) * 
                                   (large_grid - 1)/(mxw - mnw) + 
                                   0.5, 0).astype(int)
    
    # for each pixel in large grid mark position
    for i in range(n):
        mapper[position[i,TOP]:position[i,BOTTOM], 
               position[i,LEFT]:position[i,RIGHT],
               i] = 1
    
    # summarise large grid to small grid, so percent has a value
    # between 0 and 100 for the number of cells occupied for each word
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            proportions = np.sum(mapper[i * 10:(i+1) * 10,
                                        j * 10:(j+1) * 10,
                                        :],(0,1))
            proportion = np.max(proportions)
            if proportion > 20: # 20% deminimis proportion to color cell
                grid[i,j] = np.argmax(proportions)
            
    return grid

if __name__ == '__main__':
    # Save map as csv to examine later
    _, location = read_table(config.local_cfg['TEST_FILE'], config.pathtable)
    array = get_map4(location)    
    np.savetxt(config.pathtxtarray + '/' 
               + config.local_cfg['TEST_FILE'] + '_wordmap_np.csv',
               array, delimiter = ',') 
    complete, _ = mapping_completeness(array, location)
    show_array(array) # plot on screen
    print('Mapping completeness:', complete)



#%% version 3 mapping based on mapping to single cell at the middle of words
# Current version of code
def get_position(location):
    ''' Resize location based on the GRID_SIZE to process the data 
    all entries are treated as single cells in the middle of the word location
    - may enable smaller grid size but less well represents gaps between words
    '''
    n = location.shape[0]
    position = np.zeros((n,4), dtype = int) # Grid of TOP, LEFT, BOTTOM, RIGHT 
    
    location_x = location[:,[TOPLEFTX,
                             TOPRIGHTX,
                             BOTTOMRIGHTX,
                             BOTTOMLEFTX]]
    
    location_y = location[:,[TOPLEFTY,
                             TOPRIGHTY,
                             BOTTOMRIGHTY,
                             BOTTOMLEFTY]]
    
    # scale axes separately to a square grid
    mxh = np.max(location_y) # max height
    mxw = np.max(location_x) # max width
    mnh = np.min(location_y) # min height
    mnw = np.min(location_x) # min width
    
    
    used_locs = {}  # dictionary to ensure each location is unique
    
    # left adjusted down by 2 to prevent right most items disappearing off the grid
    # Take middle location: average of bounding boxes
    position[:,TOP] = np.round((np.sum(location_y,1)/4 - mnh)* 
                                (GRID_SIZE-1)/(mxh - mnh) ,0).astype(int)
    position[:,LEFT] = np.round((np.sum(location_x,1)/4 - mnw) * 
                                (GRID_SIZE-2)/(mxw - mnw), 0).astype(int)
    
    
    # check for duplicate locations and resolve by moving affecting word 1 to the right
    for i in range(n):
        while (position[i,0], position [i,1]) in used_locs:
            position [i,1] += 1
        used_locs[(position[i,0], position [i,1])] = '' # add entry to dictionary
    
    # Generate bottom and right entries so [LEFT:RIGHT] is a single point
    position[:,BOTTOM] = position[:,TOP] + 1
    position[:,RIGHT] = position[:,LEFT] + 1

    return position

def create_map(position, location):
    ''' Create array of positions recording the word encoded at each location
        For decoding the result.  No entry is -1 as 0 is a valid index'''

    target = np.ones((GRID_SIZE,GRID_SIZE), dtype = np.int64) * -1
  
    n = location.shape[0]
    for i in range(n):  # Loop through each word
        target[position[i,TOP]:position[i,BOTTOM], 
               position[i,LEFT]:position[i,RIGHT]
               ] = i
    return target

def get_map3(location):
    ''' transform location to a grid map: an array indicating which word
    is encoded at which pixel on the grid.  -1  = no entry '''
    position = get_position(location)
    target = create_map(position, location)
    return target

if __name__ == '__main__':
    _, location = read_table(config.local_cfg['TEST_FILE'], config.pathtable)
    position = get_position(location)
    print(position[:,:])  # display if just running this cell
    print('Min',np.min(position,0))
    print('Max', np.max(position,0))
    ay = get_map(location)
    complete, _ = mapping_completeness(ay, location)
    show_array(ay) # plot on screen
    print('Mapping completeness:', complete)


#%%  Version 1 mapping based on grid size
def getposition1(location):
    ''' Resize location based on the GRID_SIZE to process the data 
    all entries are treated as blocks from TOP TO BOTTOM
    probably requires GRID SIZE to be 128 to handle overlaps'''
    n = location.shape[0]
    position = np.zeros((n,4), dtype = int) # Grid of TOP, LEFT, BOTTOM, RIGHT 
    
    # scale axes separately to a square grid
    mxh = np.max(location[:,BOTTOMRIGHTY]) # max height
    mxw = np.max(location[:,BOTTOMRIGHTX]) # max width
    mnh = np.min(location[:,TOPLEFTY]) # min height
    mnw = np.min(location[:,TOPLEFTX]) # min width    
    
    # round TOP and LEFT down, and BOTTOM and RIGHT up as these will be array indices
    position[:,TOP] = np.round((location[:,TOPLEFTY] - mnh) * 
                                GRID_SIZE/(mxh - mnh),0).astype(int)
    position[:,BOTTOM] = np.round((location[:,BOTTOMRIGHTY] - mnh) * 
                                GRID_SIZE/(mxh - mnh) + 0.5, 0).astype(int)
    position[:,LEFT] = np.round((location[:,TOPLEFTX] - mnw) * 
                                GRID_SIZE/(mxw - mnw), 0).astype(int)
    position[:,RIGHT] = np.round((location[:,BOTTOMRIGHTX] - mnw) * 
                                 GRID_SIZE/(mxw - mnw) + 0.5, 0).astype(int)
    return position

#%% Version 2 mapping to single cell based on top left.  

def getposition2(location):
    ''' Resize location based on the GRID_SIZE to process the data 
    all entries are treated as single cells at the top left of the word location
    - may enable smaller grid size but less well represents gaps between words
    '''
    n = location.shape[0]
    position = np.zeros((n,4), dtype = int) # Grid of TOP, LEFT, BOTTOM, RIGHT  
    
    # scale axes separately to a square grid
    mxh = np.max(location[:,BOTTOMRIGHTY]) # max height
    mxw = np.max(location[:,BOTTOMRIGHTX]) # max width
    mnh = np.min(location[:,TOPLEFTY]) # min height
    mnw = np.min(location[:,TOPLEFTX]) # min width
    
    used_locs = {}  # dictionary to ensure each location is unique
    
    # left adjusted down by 2 to prevent right most items disappearing off the grid
    position[:,TOP] = np.round((location[:,TOPLEFTY] - mnh)* 
                                GRID_SIZE/(mxh - mnh) ,0).astype(int)
    position[:,LEFT] = np.round((location[:,TOPLEFTX] -mnw) * 
                                (GRID_SIZE-2)/(mxw - mnw), 0).astype(int)
    
    
    # check for duplicate locations and resolve by moving affecting word 1 to the right
    for i in range(n):
        while (position[i,0], position [i,1]) in used_locs:
            position [i,1] += 1
        used_locs[(position[i,0], position [i,1])] = '' # add entry to dictionary
    
    # Generate bottom and right entries so [LEFT:RIGHT] is a single point
    position[:,BOTTOM] = position[:,TOP] + 1
    position[:,RIGHT] = position[:,LEFT] + 1

    return position
