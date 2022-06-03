#!/usr/bin/env python

from data_load import unpickle_to_raw

# Define directories where we get the spectra and where we write our translation to our vocab
scans_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/ms1_scans/'
tokens_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens_0.1/no_CLS_desc_no_rettime_10000_tokens/'

unpickle_to_raw (input_dir=scans_dir, 
                output_dir=tokens_dir, 
                num_bins=10000, # number of bins used in the binning process
                peak_filter=0.1 , # top percentage of bins considered based on the highest intensity peaks
                input_size=512,  # natural BERT size is 512 (later size reductions will be done if needed)
                CLS_token=False, # if we want to add CLS token at the beginning of each sentence
                descending_order=True, #decide if we want to sort bins by descending intensity
                add_ret_time=False) # decide if we want to include retention time at the beginning of "sentence" followed by SEP token
