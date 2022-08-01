#!/usr/bin/env python

"""
Main file for the translation of mass-spec graphs to a "language" adapted for
the application of BERT model.

Author - Enric Cristòbal Cóppulo
"""

import argparse
from data_load import unpickle_to_raw

def main(args):
    # Define directories where we get the spectra and where we write our translation to our vocab
    scans_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/ms1_scans/'
    tokens_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens_' + args.peak_filter + '/' + args.case + '_' + args.num_bins  + '_tokens/'

    unpickle_to_raw (input_dir=scans_dir, 
                    output_dir=tokens_dir, 
                    num_bins=args.num_bins, 
                    peak_filter=args.peak_filter , 
                    input_size=args.input_size,  
                    CLS_token=args.CLS_token, # if we want to add CLS token at the beginning of each sentence
                    descending_order=args.desc_order, #decide if we want to sort bins by descending intensity
                    add_ret_time=args.ret_time) # decide if we want to include retention time at the beginning of "sentence" followed by SEP token

if __name__ == '__main__':
    
    # create the parser
    parser = argparse.ArgumentParser(prog='dataset_translator.py', formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=50, width=130), 
        description='''Translation of data into a language format suitable for BERT.''')
    
    parser.add_argument('--num_bins', help='Number of bins for the creation of the vocabulary in the binning process', required=True, type=int)
    parser.add_argument('--peak_filter', help='Top percentage of bins considered based on the highest intensity peaks', required=True)
    parser.add_argument('--case', help='Combination of parameters for tokens used', required=True)
    parser.add_argument('--input_size', help='Sentence length where natural BERT size is 512 (later size reductions will be done if needed)', required=True)
    parser.add_argument('--CLS_token', help='Boolean for the inclusion of [CLS] token at the beginning', type=bool)
    parser.add_argument('--desc_order', help='Boolean for ordering of peaks by descending intensity', type=bool)
    parser.add_argument('--ret_time', help='Boolean for the inclusion of retention time of the scan with [SEP] token', type=bool)
    
    args = parser.parse_args()

    main(args)

