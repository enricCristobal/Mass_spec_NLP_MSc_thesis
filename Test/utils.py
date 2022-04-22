#!/usr/bin/env python

import torch
from torchtext.vocab import build_vocab_from_iterator

import pandas as pd
import random
import os

def get_vocab(num_bins):
    # Define vocab required for the input of our fine-tuning
    all_tokens = []
    for i in range(num_bins):
        all_tokens.append([str(i)])

    vocab = build_vocab_from_iterator(all_tokens, specials=['[CLS]', '[SEP]', '[PAD]', '[EOS]', '[MASK]'])
    return vocab


def fine_tune_ds(fine_tune_files, class_param = 'Group2', labels_path = '/home/projects/cpr_10006/projects/gala_ald/data/clinical_data/ALD_histological_scores.csv'):

    # Get dir of the '.raw' extension of file_tune_names to get a match with the csv dataframe later
    fine_tune_files = [file_name[:-4] for file_name in fine_tune_files]
    labels_df = pd.read_csv(labels_path, usecols=["File name", class_param])
    #print(labels_df['File name'][0])
    #if labels_df.index[labels_df['File name'] == '[473] 20190615_QE10_Evosep1_P0000005_LiNi_SA_Plate6_A2.htrms.PG.Quantity']:
    labels_df = labels_df[labels_df[class_param] != 'QC']
    
    # Due to some data format, we need to get rid of the first 4 characters of each File_name for later matching purposes with proper file name
    beginning_file_name = labels_df['File name'].str.find(']') + 2
    for i in range(len(labels_df)):
        labels_df['File name'].iloc[i] = labels_df['File name'].iloc[i][beginning_file_name.iloc[i]:-18] 
    
    labels_df['class_id'] = labels_df[class_param].factorize()[0]
    
    # If we want to save the original class name and know its id
    #category_id_df = df[[class_param, 'class_id']].drop_duplicates()
    #category_to_id = dict(category_id_df.values)
    
    labels_df = labels_df.loc[labels_df['File name'].isin(fine_tune_files)]

    return labels_df


flatten_sample = lambda sample,dimension:[spectra for spectra in sample[dimension]]

def get_finetune_batch(sample_data_ds, batchsize, same_sample: bool): 
    # Used for fine-tuning when both mixing spectra from different samples under the same batch and not (same_sample boolean)
    # We consider the input is already just one sample
    input_data = flatten_sample(sample_data_ds, 0)
    labels_data = flatten_sample(sample_data_ds, 1)

    num_batches = len(input_data) // batchsize
    last_batch = len(input_data) % batchsize

    print('Num batches: ', num_batches, last_batch)
    print(len(input_data))
    # We do same process of randomising a bit, so we don't follow the retention times of the experiment in order
    input_batch = []; labels_batch = []; 
    indices = list(range(len(input_data)))

    if not same_sample:
        for shuffling_times in range(10):
            random.shuffle(indices)

    for batch in range(num_batches + 1) :
        batch_indices = []
        if batch < num_batches:
            for _ in range(batchsize):
                batch_indices.append(indices.pop())
        else:
                batch_indices = indices
                #print(batch_indices)

        input_batch.append(torch.stack([input_data[index] for index in batch_indices]).t())
        labels_batch.append(torch.stack([labels_data[index] for index in batch_indices]))


    return input_batch, labels_batch, num_batches

# Get datasets:
def get_labels(df, samples_list):
    # Function to get the label for each sample (file) that is part of a list of interest with file names within a df
    labels = []; out_samples = []
    for i,f in enumerate(samples_list):
        if f[:-4] not in df.values: # In case some of the samples selected are QC
            out_samples.append(f) # so there are no mathing errors when getting val_finetune_ds because a label has been removed
        else:
            index = df.index[df['File name'] == f[:-4]]
            labels.append(df.at[index[0], 'class_id'])
    for removed_sample in out_samples: # Although it seems dummy, removing straight messed up the for loop
        samples_list.remove(removed_sample)
    return labels, samples_list

# Divide samples into train and validation by specifying the percentage for each
def divide_train_val_samples(files_dir, train_perc, val_perc):

    samples_list = os.listdir(files_dir)

    limit_train = int(len(samples_list) * train_perc)
    limit_val = int(len(samples_list) * val_perc)

    train_samples = samples_list[:limit_train]
    val_samples = samples_list[limit_train:(limit_train + limit_val)]

    return train_samples, val_samples