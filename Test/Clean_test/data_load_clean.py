#!/usr/bin/env python

import pickle
import os
import numpy as np
import random
import pandas as pd

import torch
from torchtext.data.utils import get_tokenizer


""" In this file, the whole data pre-processing transformation from pickle files to raw data files with the desired input format,
as well as the final DataLoaders with all the required tensors (inputs, targets, masks, labels, etc.) ready to be passed to our 
models will be done."""


# DATA VISUALIZATION AS DF

def unpickle_to_df(pickle_dir, output_dir):
    """Function to be able to get the data from computerome as csv format to
    be able to use the visualize it as a dataframe in jupyter notebook. 
    
    Parameters:
    - pickle_dir: Directory where the pickle files are found.
    - output_dir: Directory where the csv file will be saved.

    Comment: Dummy fucntion just used to ensure a pattern on the data to avoid 
    the 10 first scans before doing the 3 rows jump to remove the "magic" scans."""

    for filename in os.listdir(pickle_dir):
        x = open(pickle_dir + filename, 'rb')
        x = pickle.load(x)
        output_file = open(output_dir + filename[:-4], 'w')
        x.to_csv(output_file)
        output_file.close()

# DATA TOKENIZATION

def unpickle_to_raw(input_dir, output_dir, num_bins, peak_filter:float = 0.1, \
    input_size: int = 512,  CLS_token: bool = True, add_ret_time: bool = True, \
    descending_order: bool = False):

    """Function to get all pickle files from input_dir to convert to raw data files 
    with bin numbers on the output_dir with the desired format.
    
    Parameters:
    - input_dir: Directory where the pickle files are found.
    - output_dir: Directory where raw data files will be saved.
    - num_bins: Number of m/z bins used to divide each spectra.
    - peak_filter: Proportion of highest intensity peaks kept.
    - input_size: Maximum "sentence" length allowed for the input and target of the model.
    - CLS_token: Boolean to choose if we want to add [CLS] token at the beginning of the
    sentence.
    - add_ret_time: Boolean for adding retention time to the input sentence.
    - descending_order: Boolean to order our peaks on intensity.
    """

    # Parse scans
    # Set up bins (from 299.9 to 1650 due to manual inspection of spectra)

    bins = np.linspace(299.9, 1650, num=num_bins)
    
    for filename in os.listdir(input_dir): 
        if filename != '20190525_QE10_Evosep1_P0000005_LiNi_SA_Plate3_A7.raw.pkl': # Giving issues after manual inspection
            x= pickle.load(open(input_dir + filename, "rb"))
            output_file = open(output_dir + filename[:-4], "w")  # [:-4] because otherwise ".pkl" extension would be added

            token_list = []
            x = x.iloc[9::3, [1,5,6]] # we get rid of the "algorithm warmup" and "magic" scans (obtained when visualizing df obtained from unpickle_to_df)
            for i in range(len(x)):
                mass_spec_scan = x.iloc[i,0] # get scan number (but we lose specific retention time) (column 0 would be ret_time)
                # but adding retention time would make vocab definition as complex as if we added intensities --> TODO!!: Consider the 2D embedding in continuous space from Henry's poster
                mz_array = x.iloc[i,1]
                inten_array = x.iloc[i,2]

                # Initialize token list including the [CLS] token and retention time as well as some special tokens
                cls_token = ['[CLS]']
                ret_time = [str(mass_spec_scan), '[SEP]']
                ret_time_spec = []

                if CLS_token:
                    ret_time_spec.extend(cls_token)
                
                if add_ret_time:
                    ret_time_spec.extend(ret_time)
                
                # skip if it starts with 900 m/z as this is an artefact in the machine
                if mz_array[0] < 900:
                    # masking out the lower peaks' proportion ( !!TODO: at some point calculate mass over charge ratio)
                    if peak_filter != 1:
                        threshold = np.quantile(inten_array, np.array([1-peak_filter]))[0]
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
                        
                    ##TODO!: Add intensities in the "sentence"? # Tricky to create voabulary of intensities

                    tokens_string =  [str(int) for int in mz_token_array]
                    
                # Collect
                ret_time_spec.extend(tokens_string)
                
                # Maximum input size for BERT is 512 
                if len(ret_time_spec) > (input_size - 1): # -1 as '[EOS]' token will be added later 
                    ret_time_spec = ret_time_spec[:(input_size -1)] 
                # Some of the scans will have more than 512 "peaks"/bins, while others will have less --> padding is needed later
                
                # Add [EOS] token to allow model differentiate between [SEP] for ret_time and end of scan
                ret_time_spec.extend(['[EOS]']) # TODO!!: Really relevant to add [EOS] token if not used during training?
            
                # Save to raw data text file
                out_string = ' '.join(ret_time_spec)
                token_list.append(out_string)

            for mass_spec in token_list:
                output_file.write('%s\n' % mass_spec)      
            output_file.close()


## DATA PRE_PROCESSING FUNCTIONS prior to model usage

def divide_train_val_samples(files_dir, train_perc, val_perc):
    """Divide samples into train and validation samples by specifying the percentage for each.

    Parameters: 
    - Files_dir: Directory where raw data files are found.
    - Train_perc: Percentage of samples in specified directory that will be used for training.
    - Val_perc: Percentage of samples in specified directory that will be used for validation."""

    samples_list = os.listdir(files_dir)

    limit_train = int(len(samples_list) * train_perc)
    limit_val = int(len(samples_list) * val_perc)

    train_samples = samples_list[:limit_train]
    val_samples = samples_list[limit_train:(limit_train + limit_val)]

    return train_samples, val_samples


## TRAINING PART

def create_training_dataset(sample, vocab, samples_dir, num_bins, \
     CLS_token: bool = True, input_size: int = 512):
    """Function to process the raw data from text files with peak numbers to the tensors that 
    will be passed as input, target, mask and labels, as part of the DataLoaders.
    This includes padding and masking.

    Parameters:
    - sample: Which sample --> file will be processed.
    - vocab: The specific vocabulary of our model. Default: will be defined
    in the main code but will be the number of bins plus some special tokens.
    - samples_dir: Directory where the files are found. 
    - num_bins: Number of m/z bins used to divide each spectra. (Used for the masking)
    - cls_token: Boolean to define the masking system and avoid masking any special character.
    - input_size: Maximum "sentence" length allowed for the input and target of the model. In 
    this case used for padding those sentences that aren't long enough."""

    # helper function to pad
    def pad_tokens(sentence, pad='[PAD]'):
        '''Pad list to input_size and get mask accordingly.'''
        mask = []
        if len(sentence) < input_size:
            pad_len = input_size-len(sentence)
            mask.extend([0]*len(sentence))
            sentence.extend([pad]*pad_len)
            mask.extend([1]*pad_len)
        else:
            mask.extend([0]* (input_size))
        return sentence, mask
    
    def yield_tokens(sample):
        tokenizer = get_tokenizer(None) # simple splitting of tokens (mostly bin numbers)
        fh = open(samples_dir + sample, 'r')
        for line in fh:
                token_list = tokenizer(line.rstrip())
                tokens_pad, mask = pad_tokens(token_list)
                yield tokens_pad, tokens_pad, mask

    # Generate tokens and pad if needed
    input_iter = []; target_iter = []; att_mask_iter = []
    for input_yield, target_yield, att_mask_yield in yield_tokens(sample):
        input_iter.append([input_yield])
        target_iter.append([target_yield])
        att_mask_iter.append([att_mask_yield])
    
    input = [item for sublist in input_iter for item in sublist]    
    target = [vocab(item) for sublist in target_iter for item in sublist]
    attention_mask = [item for sublist in att_mask_iter for item in sublist]

    # convert to list of tensors    
    target_tensor_list = [torch.tensor(L, dtype=torch.int64) for L in target]
    attention_mask_tensor = [torch.tensor(L, dtype=torch.bool) for L in attention_mask]

    # helper function to generate masked input
    def mask_input(input, CLS_token):
        j = 3 if CLS_token else 2 # not masking [CLS] token if present
        masked_indices = []
        for i in range(len(input)-(j+1)): # To not mask the ret_time nor the [EOS], [SEP] tokens
            if input[i+j] != '[PAD]':
                if random.random() < 0.15: # 15% tokens will be masked
                    second_random = random.random()
                    if second_random < 0.8: # of the 15%, 80% will be changed to [MASK] token
                        input[i+j] = '[MASK]' 
                        masked_indices.append(i+j) # save masked positions that will be used later when training the model
                    elif 0.8 <= second_random < 0.9: # of the 15%, 10% is changed for a random value of the vocab
                        input[i+j] =  str(random.randint(0,(num_bins - 1)))  
                        masked_indices.append(i+j) 
                    else:
                        masked_indices.append(i+j) # last 10% of the 15% is left unchanged           
        yield input, masked_indices

    final_input = []; mask_indices = []
    for input_line in input:
        for input, masked_indices in mask_input(input_line, CLS_token):
            final_input.append([input])
            mask_indices.append(masked_indices)
    
    input = [vocab(item) for sublist in final_input for item in sublist]
    
    # convert to list of tensors
    input_tensor_list = [torch.tensor(L, dtype=torch.int64) for L in input]
    
    return input_tensor_list, target_tensor_list, attention_mask_tensor, mask_indices


def TrainingBERTDataLoader(files_dir, vocab, num_bins: int, training_percentage: float, validation_percentage: float, CLS_token: bool):
    "Simplify 'DataLoader' creation for training BERT in one function"

    # Divide samples in training and validation sets
    train_samples, val_samples = divide_train_val_samples(files_dir, train_perc=training_percentage, val_perc=validation_percentage)

    # Create dataloaders considering the presence or not presence of CLS tokens for the masking, as well as the limited seq_length for 90% of training as mentioned in BERT paper
    # (all input_ds, target_ds and att_mask_ds will have size #spectra x 512 --> each spectra is an input to train our model)
    train_ds = [create_training_dataset(f,vocab, samples_dir = files_dir, num_bins=num_bins, CLS_token=CLS_token) for f in train_samples]
    # Create validation dataloader
    val_ds = [create_training_dataset(f,vocab, samples_dir = files_dir, num_bins=num_bins, CLS_token=CLS_token) for f in val_samples]

    return train_ds, val_ds


#FINE-TUNING PART

def fine_tune_ds(fine_tune_files, class_param = 'Group2',\
     labels_path = '/home/projects/cpr_10006/projects/gala_ald/data/clinical_data/ALD_histological_scores.csv'):

    """Function that will return a dataframe containing the samples with their file name as first column,
    The second column will be the class_param that we are considering for the classification (Group 2 by default --> Healthy vs ALD),
    and the last one is the factorization of the strings for class_param to numbers.

    Parameters:
    - Fine_tune_files: Files/Samples taht will be used for the fine-tuning.
    - Class_param: Parameter that will be used for the classification later.
    - Labels_path: Pathway to the file where data about samples is saved."""

    # Get rid of the '.raw' extension of file_tune_names to get a match with the csv dataframe later
    fine_tune_files = [file_name[:-4] for file_name in fine_tune_files]
    labels_df = pd.read_csv(labels_path, usecols=["File name", "Groups", class_param])

    labels_df = labels_df.loc[labels_df['File name'].isin(fine_tune_files)] # Consider only those we are interested in 
    labels_df = labels_df[labels_df["Groups"] != 'QC'] # Get rid of the quality controls
    
    # Due to some data format, we need to get rid of the first 4 and last 18 characters of each File_name for later matching purposes with proper file name
    beginning_file_name = labels_df['File name'].str.find(']') + 2
    for i in range(len(labels_df)):
        labels_df['File name'].iloc[i] = labels_df['File name'].iloc[i][beginning_file_name.iloc[i]:-18] 
    
    labels_df['class_id'] = labels_df[class_param].factorize()[0]
    
    # If we want to save the original class name and know its id
    #category_id_df = df[[class_param, 'class_id']].drop_duplicates()
    #category_to_id = dict(category_id_df.values)

    return labels_df


# Get datasets:

def get_labels(df, samples_list):
    """Function to get the label for each sample that is part of a list of interest with file names within the labels_df.

    Parameters:
    - df: Dataframe where labels for each file/sample is found.
    - samples_list: List of samples to consider."""

    labels = []; out_samples = []
    for f in samples_list:
        if f[:-4] not in df.values: # In case some of the samples selected are QC
            out_samples.append(f) # so there are no matching errors when getting finetune_ds because a sample has been removed
        else:
            index = df.index[df['File name'] == f[:-4]]
            labels.append(df.at[index[0], 'class_id'])
    for removed_sample in out_samples: # Although it seems dummy, removing straight messed up the for loop
        samples_list.remove(removed_sample)
    return labels, samples_list


def create_finetuning_dataset(sample, vocab, samples_dir, label, input_size: int = 512):
    """Function to process the raw data from text files with peak numbers to the tensors that 
    will be passed as input and target, as part of the DataLoaders.
    This is for the classification task.

    Parameters:
    - sample: Which sample --> file will be processed.
    - vocab: The specific vocabulary of our model. Default: will be defined
    in the main code but will be the number of bins plus some special tokens.
    - samples_dir: Directory where the files are found. 
    - label: Label of the given sample obtained from labels_df.
    - input_size: Maximum "sentence" length allowed for the input of the model. In 
    this case used for padding those sentences that aren't long enough."""
    # helper function to pad
    def pad_tokens(sentence, pad='[PAD]'):
        '''Pad list to input_size.'''
        if len(sentence) < input_size:
            pad_len = input_size-len(sentence)
            sentence.extend([pad]*pad_len)
        return sentence
    
    def yield_tokens(sample):
        tokenizer = get_tokenizer(None) # simple splitting of tokens (mostly bin numbers)
        fh = open(samples_dir + sample, 'r')
        for line in fh:
                token_list = tokenizer(line.rstrip())
                tokens_pad = pad_tokens(token_list)
                yield tokens_pad

    # generate list (tokens)
    data_iter = [yield_tokens(sample)]
    
    # Generate tokens and pad if needed
    flat_list = [item for sublist in data_iter for item in sublist]
    input_list = [vocab(item) for item in flat_list]
    
    # convert to list of tensors
    input_tensor_list = [torch.tensor(L, dtype=torch.int64) for L in input_list]

    # Get the target tensor (meaning the class_id for this sample linked to each scan of the sample)
    
    label_tensor_list = [torch.tensor(label, dtype=torch.int64) for _ in range(len(input_tensor_list))]
    
    return input_tensor_list, label_tensor_list


def FineTuneBERTDataLoader(files_dir: str, vocab, training_percentage: float, validation_percentage: float, class_param: str = 'Group2', \
    labels_path: str = '/home/projects/cpr_10006/projects/gala_ald/data/clinical_data/ALD_histological_scores.csv'):
    "Simplify 'DataLoader' creation for fine-tuning BERT in one function."

    train_samples, val_samples = divide_train_val_samples(files_dir, train_perc=training_percentage, val_perc=validation_percentage)
    finetune_samples = train_samples + val_samples

    labels_df = fine_tune_ds(finetune_samples, class_param, labels_path)
    #print(labels_df)

    # TODO!!: Issues regarding imbalanced dataset!!! Could be fixed using weight in cross entropy loss

    #print(len(labels_df.index[labels_df['class_id'] == 0]))
    #print(len(labels_df.index[labels_df['class_id'] == 1]))

    num_labels = len(labels_df['class_id'].unique())
    #print('Num labels', num_labels)

    train_labels, train_finetune_samples = get_labels(labels_df, train_samples)
    val_labels, val_finetune_samples= get_labels(labels_df, val_samples)

    train_finetune_ds = [create_finetuning_dataset(f, vocab, files_dir, train_labels[i]) for i,f in enumerate(train_finetune_samples)]
    val_finetune_ds = [create_finetuning_dataset(f, vocab, files_dir, val_labels[i]) for i,f in enumerate(val_finetune_samples)]

    return train_finetune_ds, val_finetune_ds, num_labels

            
