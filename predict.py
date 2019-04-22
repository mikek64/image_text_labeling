# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 18:12:22 2018

File to predict an invoice based on an image.
To be called from the command line

Invoices must be in the image folder for now

@author: mike_
"""

import argparse
import main
#%%

def parse_args():
    ''' Parse command line args to start generic ai '''

    parser = argparse.ArgumentParser(description='Tag invoice parts from image')
    
    # load and save options
    parser.add_argument("file",  help = "File name.  Without path looks in image subfolder.  Default .png")

    parser.add_argument("-s", "--save",  help = "Save response", action = "store_true")
    parser.add_argument("-p", "--savepath",  help = "Save path for response")    
    parser.add_argument("-r","--response", help = "Display response", action = "store_true")
    parser.add_argument("-v","--verbose", help = "Show progress", action = "store_true")
    parser.add_argument("-a","--assigned", help = "Display assigned tags", action = "store_true")
    
    args = parser.parse_args()
    
    return args

#%%

def process(args):
    ''' Start processing ''' 
    
    main.predict_an_invoice(file = args.file,
                            verbose = args.verbose,
                            save = args.save or not args.savepath is None,
                            path = args.savepath,
                            print_response = args.response,
                            print_assigned = args.assigned)
    
if __name__ == "__main__":
    args = parse_args()
    process(args)



