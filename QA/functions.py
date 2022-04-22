#!/usr/bin/env python

import string
import torch
from torch import nn, Tensor
from torchtext.vocab import build_vocab_from_iterator

import pandas as pd
import random
import os
import numpy as np
import umap.umap_ as umap
import seaborn as sns
import matplotlib.pyplot as plt


def get_vocab(num_bins):
    # Define vocab required for the input of our fine-tuning
    all_tokens = []
    for i in range(num_bins):
        all_tokens.append([str(i)])

    vocab = build_vocab_from_iterator(all_tokens, specials=['[CLS]', '[SEP]', '[PAD]', '[EOS]', '[MASK]'])
    return vocab

## DATA PRE_PROCESSING FUNCTIONS prior to model usage

def divide_train_val_samples(files_dir, train_perc, val_perc):
    # Divide samples into train and validation by specifying the percentage for each

    samples_list = os.listdir(files_dir)

    limit_train = int(len(samples_list) * train_perc)
    limit_val = int(len(samples_list) * val_perc)

    train_samples = samples_list[:limit_train]
    val_samples = samples_list[limit_train:(limit_train + limit_val)]

    return train_samples, val_samples


flatten_list = lambda tensor, dimension:[spectra for sample in tensor for spectra in sample[dimension]]
flatten_sample = lambda sample,dimension:[spectra for spectra in sample[dimension]]


def get_batch(data_ds, batchsize, epoch_status, limited_seq_len, shorter_len_perc):
    # Create function to divide input, target, att_mask for each dataloader

    # Transpose by .t() method because we want the size [seq_len, batch_size]
    # This function returns a list with len num_batches = len(input_data) // batchsize
    # where each element of this list contains a tensor of size [seq_len, batch_size]
    # Thus, looping through the whole list with len num_batches, we are going through 
    # the whole dataset, but passing it to the model in tensors of batch_size.

    # epoch status sera un numero que vindra de round(epich_actual / total_epochs) per fer lo de la 
    # len diferent de 128 a 512 per a aaccelerar el training

    input_data = flatten_list(data_ds, 0)
    target_data = flatten_list(data_ds, 1)
    att_mask = flatten_list(data_ds, 2)
    labels = flatten_list(data_ds, 3)
    
    num_batches = len(input_data) // batchsize
    input_batch = []; target_batch = []; att_mask_batch = []; labels_batch = []
    indices = list(range(len(input_data)))

    for shuffling_times in range(10):
        random.shuffle(indices)

    for batch in range(num_batches):
        batch_indices = []

        for j in range(batchsize):
            batch_indices.append(indices.pop())

        if epoch_status <= shorter_len_perc:
            input_batch.append(torch.stack([input_data[index][:limited_seq_len] for index in batch_indices]).t())
            target_batch.append(torch.stack([target_data[index][:limited_seq_len] for index in batch_indices]).t())
            att_mask_batch.append(torch.stack([att_mask[index][:limited_seq_len] for index in batch_indices])) # because size passed needs to be (N, S), being N batchsize and S seq_len
            # Just get those smaller than limited_seq_len
            smaller_labels = np.array([labels[index] for index in batch_indices])
            labels_batch.append(smaller_labels[smaller_labels<=limited_seq_len].tolist())
        else:
            input_batch.append(torch.stack([input_data[index] for index in batch_indices]).t())
            target_batch.append(torch.stack([target_data[index] for index in batch_indices]).t())
            att_mask_batch.append(torch.stack([att_mask[index] for index in batch_indices]))
            labels_batch.append([labels[index] for index in batch_indices])

    return input_batch, target_batch, att_mask_batch, labels_batch, num_batches


def fine_tune_ds(fine_tune_files, class_param = 'Group2',\
     labels_path = '/home/projects/cpr_10006/projects/gala_ald/data/clinical_data/ALD_histological_scores.csv'):
    '''
    Function that will return a dataframe containing the samples with their file name as first column,
    The second column will be the class_param that we are considering for the classification (Group 2 by default),
    and the last one is the factorization of the strings for class_param to numbers.
    '''
    # Get dir of the '.raw' extension of file_tune_names to get a match with the csv dataframe later
    fine_tune_files = [file_name[:-4] for file_name in fine_tune_files]
    labels_df = pd.read_csv(labels_path, usecols=["File name", class_param])
    
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


def get_finetune_batch(sample_data_ds, batchsize, same_sample: bool): 
    # Used for fine-tuning when both mixing spectra from different samples under the same batch and not (same_sample boolean)
    # We consider the input is already just one sample
    flatten = flatten_sample if same_sample == True else flatten_list
    input_data = flatten(sample_data_ds, 0)
    labels_data = flatten(sample_data_ds, 1)
    
    num_batches = len(input_data) // batchsize
    last_batch = len(input_data) % batchsize

    num_batches_final = num_batches + 1 if last_batch else num_batches
    
    # We do same process of randomising a bit, so we don't follow the retention times of the experiment in order
    input_batch = []; labels_batch = []; 
    indices = list(range(len(input_data)))

    if not same_sample:
        for shuffling_times in range(10):
            random.shuffle(indices)

    for batch in range(num_batches_final) if last_batch else range(num_batches):
        batch_indices = []
        if batch < num_batches:
            for _ in range(batchsize):
                batch_indices.append(indices.pop())
        else:   
                batch_indices = indices

        input_batch.append(torch.stack([input_data[index] for index in batch_indices]).t())
        labels_batch.append(torch.stack([labels_data[index] for index in batch_indices]))

    return input_batch, labels_batch, num_batches_final

# Get datasets:
def get_labels(df, samples_list):
    # Function to get the label for each sample (file) that is part of a list of interest with file names within a df
    labels = []; out_samples = []
    for i,f in enumerate(samples_list):
        if f[:-4] not in df.values: # In case some of the samples selected are QC
            out_samples.append(f) # so there are no matching errors when getting val_finetune_ds because a label has been removed
        else:
            index = df.index[df['File name'] == f[:-4]]
            labels.append(df.at[index[0], 'class_id'])
    for removed_sample in out_samples: # Although it seems dummy, removing straight messed up the for loop
        samples_list.remove(removed_sample)
    return labels, samples_list


## MODEL USAGE FUNCTIONS for training, evaluation and plotting results

def plot_embeddings(model: nn.Module, samples_names: list, dataset: list, batchsize: int, \
                    aggregation_layer: string, plot_dir: string, device: torch.device):

    assert aggregation_layer in ["sample", "ret_time"], "aggregation_layer must be either at 'sample' level or 'ret_time' level"

    model.eval()
    reducer = umap.UMAP(n_neighbors=2)
    dataframe = []

    for sample in range(len(dataset)):
        #print('Sample number: ', sample + 1)
        inputs, class_labels, num_batches = get_finetune_batch(dataset[sample], batchsize, same_sample=True)
        
        hidden_vectors_sample = []
        for batch in range(num_batches):
            #if batch == int(num_batches / 2):
                #print('Half sample!') 
            hidden_vectors_batch = model(inputs[batch].to(device)) # torch.Tensor of size [batchsize x embed_size]
            hidden_vectors_sample.append(hidden_vectors_batch.tolist())

        flat_hidden_vectors_sample = [scan for batched_scans in hidden_vectors_sample for scan in batched_scans]

        samples_and_time = list(map(list,zip([samples_names[sample]]*len(dataset[sample][0]), range(len(dataset[sample][0])), [class_labels[0][0].tolist()]*len(dataset[sample][0]))))
        hidden_vectors = flat_hidden_vectors_sample
        dataframe.append(list(map(list.__add__, samples_and_time, hidden_vectors)))
        
    flat_dataframe = [sample_scan for sample_dataframe in dataframe for sample_scan in sample_dataframe]
    df = pd.DataFrame(flat_dataframe, columns = ['Sample_name', 'Scan_order', 'Label'] + list(range(hidden_vectors_batch.size()[1])))

    if aggregation_layer == "sample":
        df = df.groupby('Sample_name').mean()

    embedding = reducer.fit_transform(df)
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df.Label)
    plt.legend(handles=scatter.legend_elements()[0], labels = ['Healthy', 'ALD'], title="Patients status")
    plt.title('UMAP projection of the embedding space by patients', fontsize=14)
    plt.savefig(plot_dir)
      



        
