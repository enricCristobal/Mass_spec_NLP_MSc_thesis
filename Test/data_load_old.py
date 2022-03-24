#!/usr/bin/env python

import pickle
import os
import numpy as np
import random

import torch
from torch import Tensor
from typing import Tuple

from torchtext.data.utils import get_tokenizer


class data_load:
    """ Wtih this class the whole process of transformation from pickle 
    files to raw data files as well as the preparation until the final 
    dataloader ready to be passed to our model will be done.

    Parameters:
    - output_dir: Directory where raw data files will be saved."""

    def __init__(self, output_dir, num_bins, peak_filter = 0.1, input_size = 512):
        self.output_dir = output_dir
        self.num_bins = num_bins
        self.peak_filter = peak_filter
        self.input_size = input_size 
    
    def pickle_to_raw(self, input_dir, add_ret_time = True, descending_order = False):
        """Function to get all pickle files from input_dir to convert
        to raw data files with bin numbers on the output_dir.
        
        Parameters:
        - input_dir: Directory where pickle files are found.
        - num_bins: Number of bins used to parse spectra.
        - add_ret_time: Boolean to add retention time.
        - peak_filter: Proportion highest peaks we want to keep.
        - descending_order: Boolean to order our peaks on intensity."""

        # Parse scans
        # Set up bins (from 299.9 to 1650 due to manual inspection of spectra)

        bins = np.linspace(299.9, 1650, num=self.num_bins)

        for filename in os.listdir(input_dir):
            x= pickle.load(open(input_dir + filename, "rb"))
            output_file = open(self.output_dir + filename[:-4], "w") 
            # [:-4] because otherwise ".pkl" extension would be added

            # Initialize counters and empty lists
            bad = 0; good = 0; threshholds_list = []; token_list = []

            for i in range(len(x)):
                #ret_time = x.iloc[i,0]
                mass_spec_scan = x.iloc[i,1] # get scan number (but we lose specific retention time)
                # but adding retention time makes vocab definition as complex as if we added intensities
                mz_array = x.iloc[i,5]
                inten_array = x.iloc[i,6]

                # Initialize token list including the retention time as well as some special tokens
                
                if add_ret_time:
                    ret_time_spec = [str(mass_spec_scan), '[SEP]']
                else:
                    ret_time_spec = []
                
                # skip if it starts with 900 m/z as this is an artefact in the machine
                if mz_array[0] > 900:
                    bad += 1
                    continue
                else:
                    good += 1
                    # masking out the lower peaks' proportion
                    if self.peak_filter != 1:
                        threshold = np.quantile(inten_array, np.array([1-self.peak_filter]))[0]
                        threshholds_list.append(threshold)
                        mask = np.ma.masked_where(inten_array<threshold, inten_array) # mask bottom 90% peaks that are below the threshold
                        mz_masked_array = np.ma.masked_where(np.ma.getmask(mask), mz_array)
                        inten_masked_array = np.ma.masked_where(np.ma.getmask(mask), inten_array)
                        mz_flt_array = mz_array[mz_masked_array.mask==False]
                        inten_flt_array = inten_array[inten_masked_array.mask==False]
                    else:
                        mz_flt_array = mz_array
                        inten_flt_array = inten_array

                    # discretize to tokens (e.g. bin number)
                    mz_token_array = np.digitize(mz_flt_array, bins)

                    # Order bin numbers by desc
                    if descending_order:
                        zipped_arrays = zip(inten_flt_array, mz_token_array)
                        sorted_arrays = sorted(zipped_arrays, reverse = True)  
                        mz_token_array = [element for _,element in sorted_arrays]
                        
                    ##TODO!: Add intensities in the "sentence"? 
                    # Tricky to create voabulary of intensities

                    tokens_string =  [str(int) for int in mz_token_array]
                    
                # Collect
                ret_time_spec.extend(tokens_string)
                
                # Maximum input size for BERT is 512
                if len(ret_time_spec) > (self.input_size - 1):
                    ret_time_spec = ret_time_spec[:(self.input_size -1)] 
                
                # Add [EOS] token to allow model differentiate between [SEP] for ret_time and end of scan
                ret_time_spec.extend(['[EOS]']) 
               
                # Save to raw data text file
                out_string = ' '.join(ret_time_spec)
                token_list.append(out_string)
                
                #token_list.append(ret_time_spec)

            for mass_spec in token_list:
                output_file.write('%s\n' % mass_spec)      
            output_file.close()
    
    def create_dataset(self, sample, vocab):
        """Function to process the raw data from text files with peak numbers.
        This includes padding. 
        Division between training and testing will be done in the main code."""

        # helper function to pad
        def pad_tokens(sentence, pad='[PAD]'):
            '''Pad list to at least n length and get mask accordingly'''
            mask = []
            if len(sentence) < self.input_size:
                pad_len = self.input_size-len(sentence)
                mask.extend([0]*len(sentence))
                sentence.extend([pad]*pad_len)
                mask.extend([1]*pad_len)
            else:
                mask.extend([0]* (self.input_size))
            return sentence, mask
        
        ## HOW to return these two things so we only run through the files once, but doesn't affect
        # the yield further below to get the data_iter 
        # Might get idea from here: https://stackoverflow.com/questions/16856251/yield-multiple-values

        def yield_tokens(sample):
            tokenizer = get_tokenizer(None) # simple splitting of tokens (mostly bin numbers)
            fh = open(self.output_dir + sample, 'r')
            for line in fh:
                    token_list = tokenizer(line.rstrip())
                    tokens_pad, _ = pad_tokens(token_list)
                    yield tokens_pad 

        def yield_att_mask(sample):
        #SILLY FUNCTION to get mask without affecting the yielded tokens --> FIND SOLUTION!
            tokenizer = get_tokenizer(None) # simple splitting of tokens (mostly bin numbers)
            fh = open(self.output_dir + sample, 'r')
            for line in fh:
                    token_list = tokenizer(line.rstrip())
                    _, mask = pad_tokens(token_list)
                    yield mask 

        # Generate tokens and pad if needed
        # TODO!!: Optimize iter creation
        target_iter = []
        target_iter.append(yield_tokens(sample))

        input_iter = []
        input_iter.append(yield_tokens(sample))

        att_mask_iter = []
        att_mask_iter.append(yield_att_mask(sample))

        
        # convert into vocabulary
        input = [item for sublist in input_iter for item in sublist]
        target = [vocab(item) for sublist in target_iter for item in sublist]
        attention_mask = [item for sublist in att_mask_iter for item in sublist]
        # convert to list of tensors
        
        target_tensor_list = [torch.tensor(L, dtype=torch.int64) for L in target]
        attention_mask_tensor = [torch.tensor(L, dtype=torch.bool) for L in attention_mask]
    
        # helper function to generate masked input
        def mask_input(src): #!! This way we might be masking padding tokens --> must be changed
            input = src
            for i in range(len(src)):
                if input[i] != '[PAD]':
                    if random.random() < 0.15:
                        second_random = random.random()
                        if second_random < 0.8:
                            input[i] = '[MASK]'
                        elif 0.8 <= second_random < 0.9:
                            input[i] =  str(random.randint(0,49999))               
            yield input

        input = [mask_input(input_line) for input_line in input]  
        input = [vocab(item) for sublist in input for item in sublist]
        
        # convert to list of tensors
        input_tensor_list = [torch.tensor(L, dtype=torch.int64) for L in input]
        
        return input_tensor_list, target_tensor_list, attention_mask_tensor


                    
            
