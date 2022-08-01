#!/usr/bin/env python

"""
Whole data pre-processing transformation from pickle files to raw data files
with the desired input format, as well as the final DataLoaders with all the 
required tensors (inputs, targets, padding masks, labels, etc.) ready to be 
passed to our models will be done.

Author - Enric Cristòbal Cóppulo
"""
import pickle
import os

import numpy as np
import random
import pandas as pd
from math import ceil, floor

import torch
from torchtext.data.utils import get_tokenizer


# DATA VISUALIZATION AS DF

def unpickle_to_df(pickle_dir, output_dir):
    """
    Function to be able to get the data from computerome as csv format to
    be able to visualize it as a dataframe in jupyter notebook. 
    
    Parameters:
    - pickle_dir: Directory where the pickle files are found.
    - output_dir: Directory where the csv file will be saved.'

    Comment: Dummy fucntion just used to ensure a pattern on the data to avoid 
    the 10 first scans before doing the 3 rows jump to remove the "magic" scans.
    """

    for filename in os.listdir(pickle_dir):
        x = open(pickle_dir + filename, 'rb')
        x = pickle.load(x)
        output_file = open(output_dir + filename[:-4], 'w')
        x.to_csv(output_file)
        output_file.close()

# DATA TOKENIZATION

def unpickle_to_raw(input_dir, output_dir, num_bins, peak_filter: float = 0.1, \
    input_size: int = 512,  CLS_token: bool = True, add_ret_time: bool = True, \
    descending_order: bool = True):
    """
    Function to get all pickle files from input_dir to convert to raw data files 
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
                # but adding retention time would make vocab definition as complex as if we added intensities 
                mz_array = x.iloc[i,1]
                #TODO!: Add intensities in the "sentence"? # Tricky to create voabulary of intensities --> # --> Idea: Consider the 2D embedding in continuous space from Henry's poster
                inten_array = x.iloc[i,2]

                # Initialize token list including the [CLS] token and retention time as well as some special tokens
                ret_time_spec = []
                if CLS_token:
                    ret_time_spec.extend(['[CLS]'])
                
                if add_ret_time:
                    ret_time_spec.extend([str(mass_spec_scan), '[SEP]'])
                
                # skip if it starts with 900 m/z as this is an artefact in the machine
                if mz_array[0] < 900:
                    # masking out the lower peaks' proportion (!!TODO: at some point calculate mass over charge ratio)
                    if peak_filter != 1:
                        threshold = np.quantile(inten_array, np.array([1-peak_filter]))[0]
                        mask = np.ma.masked_where(inten_array<threshold, inten_array) # mask/remove bottom 90% peaks that are below the threshold
                        mz_masked_array = np.ma.masked_where(np.ma.getmask(mask), mz_array)
                        inten_masked_array = np.ma.masked_where(np.ma.getmask(mask), inten_array)
                        mz_flt_array = mz_array[mz_masked_array.mask==False]
                        inten_flt_array = inten_array[inten_masked_array.mask==False]
                    else:
                        mz_flt_array = mz_array
                        inten_flt_array = inten_array

                    # discretize to tokens (e.g. bin number) gives bin value from bin list for m/z of each peak
                    mz_token_array = np.digitize(mz_flt_array, bins)

                    # Order bin numbers by descending intensity
                    if descending_order:
                        zipped_arrays = zip(inten_flt_array, mz_token_array)
                        sorted_arrays = sorted(zipped_arrays, reverse = True)  
                        mz_token_array = [element for _,element in sorted_arrays]
                        
                    tokens_string =  [str(int) for int in mz_token_array]
                    
                # Collect bin numbers
                ret_time_spec.extend(tokens_string)
                
                # Maximum input size for BERT is 512 
                if len(ret_time_spec) > input_size: # !!(input_size - 1  if we add '[EOS]' token at the end later 
                    ret_time_spec = ret_time_spec[:input_size] #!! (input_size - 1) if '[EOS]' token added
                # Some of the scans will have more than 512 "peaks"/bins, while others will have less --> padding is needed later
                
                # Add [EOS] token to allow model differentiate between [SEP] for ret_time and end of scan
                #ret_time_spec.extend(['[EOS]'])
            
                # Save to raw data text file
                out_string = ' '.join(ret_time_spec)
                token_list.append(out_string)

            for mass_spec in token_list:
                output_file.write('%s\n' % mass_spec)      
            output_file.close()


## DATA PRE_PROCESSING FUNCTIONS prior to model usage

def divide_train_val_samples(files_dir, train_perc, val_perc):
    """
    Divide samples into train and validation samples by specifying the percentage for each.
    Parameters: 
    - Files_dir: Directory where raw data files are found.
    - Train_perc: Percentage of samples in specified directory that will be used for training.
    - Val_perc: Percentage of samples in specified directory that will be used for validation.
    """

    samples_list = os.listdir(files_dir)

    limit_train = int(len(samples_list) * train_perc)
    limit_val = int(len(samples_list) * val_perc)

    train_samples = samples_list[:limit_train]
    val_samples = samples_list[limit_train:(limit_train + limit_val)]

    return train_samples, val_samples


## TRAINING PART

def create_training_dataset(sample, vocab, samples_dir, \
     CLS_token: bool, add_ret_time: bool, input_size: int, data_repetition: int = 1):
    """
    Function to process the raw data from text files with peak numbers to the tensors that 
    will be passed as input, target, mask and labels, as part of the DataLoaders.
    This includes padding and masking.
    Parameters:
    - sample: Which sample/file will be processed.
    - vocab: The specific vocabulary of our model. Default: will be defined
    in the main code but will be the number of bins plus some special tokens.
    - samples_dir: Directory where the files are found. 
    - num_bins: Number of m/z bins used to divide each spectra. (Used for the masking)
    - cls_token: Boolean to define the masking system and avoid masking any special character.
    - add_ret_time: Boolean for adding retention time to the input sentence.
    - input_size: Maximum "sentence" length allowed for the input and target of the model. In 
    this case used for padding those sentences that aren't long enough.
    """

    # helper function to pad
    def pad_tokens(sentence, pad='[PAD]'):
        "Pad list to input_size and get padding mask accordingly."

        pad_len = input_size-len(sentence)
        padding_mask = [0]*len(sentence)
        sentence.extend([pad]*pad_len)
        padding_mask.extend([1]*pad_len)
  
        return sentence, padding_mask
    
    def yield_tokens(sample):
        tokenizer = get_tokenizer(None) # simple splitting of tokens (mostly bin numbers)
        fh = open(samples_dir + sample, 'r')
        for line in fh:
            token_list = tokenizer(line.rstrip())
            if len(token_list) < input_size:
                tokens_pad, padding_mask = pad_tokens(token_list)
            else:
                tokens_pad = token_list[:input_size]
                padding_mask = [0] * input_size
            yield tokens_pad, tokens_pad, padding_mask

    # Generate tokens and pad if needed
    input_iter = []; target_iter = []; padding_mask_iter = []
    for _ in range(data_repetition): # we go through our data samples more than once, by masking in different places to give more data to the model
        for input_yield, target_yield, padding_mask_yield in yield_tokens(sample):
            input_iter.append([input_yield])
            target_iter.append([target_yield])
            padding_mask_iter.append([padding_mask_yield])
    
    input = [item for sublist in input_iter for item in sublist] # flatten the different lists
    target = [vocab(item) for sublist in target_iter for item in sublist]
    padding_mask = [item for sublist in padding_mask_iter for item in sublist]

    # convert to list of tensors    
    target_tensor_list = [torch.tensor(L, dtype=torch.int64) for L in target]
    padding_mask_tensor = [torch.tensor(L, dtype=torch.bool) for L in padding_mask]

    # helper function to generate masked input
    def mask_input(input, CLS_token, add_ret_time):

        j = 3 if CLS_token and add_ret_time else 2 if add_ret_time else 1 if CLS_token else 0 # not masking [CLS] token if present
        
        masked_indices = []
        if '[PAD]' in input:
            last_masking_index = input.index('[PAD]')
        else: 
            last_masking_index = len(input) # - 1 in case '[EOS]' were added
        potential_masking_input = input[j:last_masking_index] # j to avoid special tokens beginning sentence
        masking_labels = random.sample(list(enumerate(potential_masking_input)), round(0.15*len(potential_masking_input))) # get 15% tokens to be masked

        if len(masking_labels) > 4:
            for component, _ in masking_labels:
                masked_indices.append(j + component)  # save masked positions that will be used later when training the model
        
            mask_labels = [masking_labels.pop(random.randrange(len(masking_labels))) for _ in range(round(0.8*len(masking_labels)))] # of the 15%, 80% will be changed to [MASK] token
            operation = random.choice((ceil,floor)) # we include this to not affect for odd values by doing either round or floor so it is a fair 50-50% for the remaining 20% of labels
            change_labels = [masking_labels.pop(random.randrange(len(masking_labels))) for _ in range(operation(0.5*len(masking_labels)))]  # of the 15%, 10% is changed for a random value of the vocab
        # last 10% of labels are for tokens kept the same
        # all_labels = [(index1, value1), (index2, value2), ...]
        # change input accordingly
            for component, _ in mask_labels: 
                input[j + component] = '[MASK]'  

            for component, _ in change_labels:
                input[j + component] = str(random.randint(5,len(vocab)-6))  # 5 becuase we have 5 special tokens
        else:
            masked_indices = False
        yield input, masked_indices
    
    final_input = []; mask_indices = []
    for input_line in input:
        for input, masked_indices in mask_input(input_line, CLS_token, add_ret_time):
            if masked_indices: # to avoid when masking is shorter than 5 tokens
                final_input.append([input])
                mask_indices.append(masked_indices)
        
    input = [vocab(item) for sublist in final_input for item in sublist]
    # convert to list of tensors
    input_tensor_list = [torch.tensor(L, dtype=torch.int64) for L in input]
        
    return input_tensor_list, target_tensor_list, padding_mask_tensor, mask_indices


def TrainingBERTDataLoader(files_dir, vocab, training_percentage: float, validation_percentage: float, CLS_token: bool,\
     add_ret_time: bool, input_size: int = 512, data_repetition: int = 1, BERT_analysis: bool = False):
    "Simplify 'DataLoader' creation for training BERT in one function"

    # Divide samples in training and validation sets
    train_samples, val_samples = divide_train_val_samples(files_dir, train_perc=training_percentage, val_perc=validation_percentage)

    if BERT_analysis == False:
    # Create dataloaders considering the presence or not presence of CLS tokens for the masking, as well as the limited seq_length for 90% of training as mentioned in BERT paper
    # (all input_ds, target_ds and att_mask_ds will have size #spectra x 512 --> each spectra is an input to train our model)
        train_ds = [create_training_dataset(f,vocab, samples_dir = files_dir, CLS_token=CLS_token,\
            add_ret_time=add_ret_time, input_size = input_size, data_repetition=data_repetition) for f in train_samples]
    else:
        train_ds = None

    # Create validation dataloader
    val_ds = [create_training_dataset(f,vocab, samples_dir = files_dir, CLS_token=CLS_token, \
        add_ret_time=add_ret_time, input_size=input_size, data_repetition=data_repetition) for f in val_samples]
    
    return train_ds, val_ds


#FINE-TUNING PART

def fine_tune_ds(fine_tune_files, class_param = 'Group2', kleiner_type = None, \
     labels_path = '/home/projects/cpr_10006/projects/gala_ald/data/clinical_data/ALD_histological_scores.csv'):
    """
    Function that will return a dataframe containing the samples with their file name as first column,
    The second column will be the class_param that we are considering for the classification (Group 2 by default --> Healthy vs ALD),
    and the last one is the factorization of the strings for class_param to numbers.
    Parameters:
    - Fine_tune_files: Files/Samples taht will be used for the fine-tuning.
    - Class_param: Parameter that will be used for the classification later.
    - kleiner_type: Differentiation for classficiation using kleiner between significant or advanced fibrosis, based on the intervals
    chosen for factorizing samples.
    - Labels_path: Pathway to the file where data about samples is saved.
    """

    assert class_param in ["Group2", "nas_inflam", "kleiner", "nas_steatosis_ordinal"], \
        "class_param allowed are 'Group2', 'nas_inflam', 'kleiner' (defining kleiner_type) and 'nas_steatosis_ordinal'"
    
    # Get rid of the '.raw' extension of file_tune_names to get a match with the csv dataframe later
    fine_tune_files = [file_name[:-4] for file_name in fine_tune_files]
    labels_df = pd.read_csv(labels_path, usecols=["File name", "Groups", class_param])
    
    # Due to some data format, we need to get rid of the first 4 and last 18 characters of each File_name for later matching purposes with proper file name
    beginning_file_name = labels_df['File name'].str.find(']') + 2
    
    for i in range(len(labels_df)):
        labels_df['File name'].iloc[i] = labels_df['File name'].iloc[i][beginning_file_name.iloc[i]:-18]

    labels_df = labels_df.loc[labels_df['File name'].isin(fine_tune_files)] # Consider only those we are interested in  
    labels_df = labels_df[labels_df["Groups"] != 'QC'] # Get rid of the quality controls

    if class_param != 'Group2':
        labels_df = labels_df[~labels_df[class_param].isnull()] # Get rid of ALD patients with no scores for the chosen parameter
    
    # Depending on the parameter, we have different binary classifications depending on the scores
    if class_param == 'Group2':
        labels_df['class_id'] = labels_df[class_param].factorize(sort=True)[0] #sort to ensure ALD is 0 --> considered positive and HP is 1 (alphabetic sorting)

    elif class_param == 'nas_inflam':
        labels_df['class_id'] = pd.cut(labels_df[class_param], bins=[-1,1,5], labels=[1,0])

    elif class_param == 'kleiner':
        if kleiner_type == 'significant':
            labels_df['class_id'] = pd.cut(labels_df[class_param], bins=[-1,1,4], labels=[1,0])
            
        elif kleiner_type == 'advanced':
            labels_df['class_id'] = pd.cut(labels_df[class_param], bins=[-1,2,4], labels=[1,0])
  
    
    elif class_param == 'nas_steatosis_ordinal':
        labels_df['class_id'] = pd.cut(labels_df[class_param], bins=[-1,0,4], labels=[1,0])
    
    # If we want to save the original class name and know its id
    #category_id_df = df[[class_param, 'class_id']].drop_duplicates()
    #category_to_id = dict(category_id_df.values)
 
    return labels_df

# Get datasets:

def get_labels(df, samples_list):
    """
    Function to get the label for each sample that is part of a list of interest with file names within the labels_df.
    Parameters:
    - df: Dataframe where labels for each file/sample is found.
    - samples_list: List of samples to consider.
    """

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


def create_finetuning_dataset(sample, vocab, samples_dir, label, min_scan_count, input_size: int = 512):
    """
    Function to process the raw data from text files with peak numbers to the tensors that 
    will be passed as input and target, as part of the DataLoaders.
    This is for the classification task.
    Parameters:
    - sample: Which sample --> file will be processed.
    - vocab: The specific vocabulary of our model. Default: will be defined
    in the main code but will be the number of bins plus some special tokens.
    - samples_dir: Directory where the files are found. 
    - label: Label of the given sample obtained from labels_df.
    - input_size: Maximum "sentence" length allowed for the input of the model. In 
    this case used for padding those sentences that aren't long enough.
    """
    
    # helper function to pad
    def pad_tokens(sentence, pad='[PAD]'):
        '''Pad list to input_size.'''
        pad_len = input_size-len(sentence)
        sentence.extend([pad]*pad_len)
        return sentence
    
    def yield_tokens(sample, min_scan_count):
        tokenizer = get_tokenizer(None) # simple splitting of tokens (mostly bin numbers)
        fh = open(samples_dir + sample, 'r')

        if min_scan_count:
            lines = fh.readlines()[:min_scan_count] # when CNN
        else:
            lines = fh.readlines()
        
        for i in range(len(lines)):
            token_list = tokenizer(lines[i].rstrip())
            if len(token_list) < input_size:
                tokens_pad = pad_tokens(token_list)
            else:
                tokens_pad = token_list[:input_size]
            yield tokens_pad

    data_iter = [yield_tokens(sample, min_scan_count)]
   
    # Generate tokens and pad if needed
    flat_list = [item for sublist in data_iter for item in sublist]
    input_list = [vocab(item) for item in flat_list]
    
    # convert to list of tensors
    input_tensor_list = [torch.tensor(L, dtype=torch.int64) for L in input_list]

    # Get the target tensor (meaning the class_id for this sample linked to each scan of the sample)
    label_tensor_list = [torch.tensor(label, dtype=torch.int64) for _ in range(len(input_tensor_list))]
    
    return input_tensor_list, label_tensor_list


def count_scans(samples_dir, samples):
    """
    Function to find the minimum number of scans found among the
    samples considered. (Used for cropping inputs to the same 
    length when needed in fine-tuning and other analysis processes.)

    Parameters:
    - samples_dir: Directory where samples' files are found.
    - samples: Samples considered for each case.
    """
    min_scan_count = float('inf')
    for sample in samples:
        fh = open(samples_dir + sample, 'r')
        scan_count = len(fh.readlines())
        if scan_count < min_scan_count:
            min_scan_count = scan_count
    return min_scan_count


def compensate_imbalance(labels_list, samples_list, n_first_class, n_second_class):

    n_bigger_class = max(n_first_class, n_second_class)
    n_smaller_class = min(n_first_class, n_second_class)
    
    if n_first_class > n_second_class:
        potential_indices = [indices for indices, label in enumerate(labels_list) if label == 1]
        potential_repeat_samples = [samples_list[index] for index in potential_indices]
        min_label = 1
    else:
        potential_indices = [indices for indices, label in enumerate(labels_list) if label == 0]
        potential_repeat_samples = [samples_list[index] for index in potential_indices]
        min_label = 0
    
    selection = random.choices(potential_repeat_samples, k=(n_bigger_class - n_smaller_class))
    
    samples_list.extend(selection)
    labels_list.extend([min_label]*(n_bigger_class - n_smaller_class))
   
    #Shuffle to avoid having all of the same minority class at the end of the datatasets
    shuffling = list(zip(samples_list, labels_list))
    for shuffling_times in range(10):
        random.shuffle(shuffling)
    samples_list, labels_list = zip(*shuffling)
    
    return samples_list, labels_list
    

def FineTuneBERTDataLoader(files_dir: str, vocab, training_percentage: float, validation_percentage: float, crop_input: bool, \
    scan_detection: bool = False, class_param: str = 'Group2', kleiner_type: str = None, \
    labels_path: str = '/home/projects/cpr_10006/projects/gala_ald/data/clinical_data/ALD_histological_scores.csv'):
    "Simplify 'DataLoader' creation for fine-tuning BERT in one function."

    train_samples, val_samples = divide_train_val_samples(files_dir, train_perc=training_percentage, val_perc=validation_percentage)
    finetune_samples = train_samples + val_samples
    labels_df = fine_tune_ds(finetune_samples, class_param, kleiner_type, labels_path)
    
    #print(len(labels_df.index[labels_df['class_id'] == 0]))
    #print(len(labels_df.index[labels_df['class_id'] == 1]))
    
    num_labels = len(labels_df['class_id'].unique())
    
    train_labels, train_finetune_samples = get_labels(labels_df, train_samples)
    val_labels, val_finetune_samples= get_labels(labels_df, val_samples)

    # Compensate for class imbalance (for now just done for a binary classification which is always our case)
    # Just compensat if one class is present more than 10% with respect to the other
    
    if crop_input: # keep all samples with same number of scans for CNN or VAE with conv
        min_scan_count = count_scans(files_dir, finetune_samples)
    else:
        min_scan_count = None

    if scan_detection == False: #To only work with the last 10% never seen by the model for scan detection withouth having to go through the other 90% creating the dataset for nothing
        train_first_class = train_labels.count(0); train_second_class = train_labels.count(1)
        val_first_class = val_labels.count(0); val_second_class = val_labels.count(1)

        if abs(train_first_class-train_second_class)/min(train_first_class, train_second_class) > 0.1:
            train_finetune_samples, train_labels = compensate_imbalance(train_labels, train_finetune_samples, train_first_class, train_second_class)
            
        if abs(val_first_class-val_second_class)/min(val_first_class, val_second_class) > 0.1:
            val_finetune_samples, val_labels = compensate_imbalance(val_labels, val_finetune_samples, val_first_class, val_second_class)
        
        train_finetune_ds = [create_finetuning_dataset(f, vocab, files_dir, train_labels[i], min_scan_count) for i,f in enumerate(train_finetune_samples)]
    else:
        train_finetune_ds = None


    val_finetune_ds = [create_finetuning_dataset(f, vocab, files_dir, val_labels[i], min_scan_count) for i,f in enumerate(val_finetune_samples)]

    return train_finetune_ds, val_finetune_ds, num_labels, min_scan_count
