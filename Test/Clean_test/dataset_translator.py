#!/usr/bin/env python

from data_load_clean import unpickle_to_raw

# Define directories where we get the spectra and where we write our translation to our vocab
#scans_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/ms1_scans/'
#tokens_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens/scan_desc_tokens_CLS/'

# LOCAL PATHWAYS!
scans_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\pickle_data\\' 
tokens_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\dummy_data\\' 

# Define needed parameters
num_bins = 50000 # number of bins used in the binning process
peak_filter = 0.1 # we'll just consider the bins with 10% highest peaks
input_size = 100 # natural BERT size (later size reductions will be done later if needed)
CLS_token = True # if we want to add CLS token at the beginning of each sentence
retention_time = True # decide if we want to include retention time at the beginning of "sentence" followed by SEP token
descending_order = True # decide if we want to sort bins by descending intensity

unpickle_to_raw (input_dir=scans_dir, output_dir=tokens_dir, num_bins=num_bins, peak_filter=peak_filter,\
    input_size=input_size, CLS_token=CLS_token, add_ret_time=retention_time, descending_order=descending_order)



