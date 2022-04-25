#!/usr/bin/env python

import string
import torch
from torch import nn, Tensor
from torchtext.vocab import build_vocab_from_iterator

import pandas as pd
import random
import os
import time
from statistics import mean
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt

# VOCABULARY DEFINITION
def get_vocab(num_bins):
    """Define vocab used for our model defining the number of bins used to bin the scans."""
    all_tokens = []
    for i in range(num_bins):
        all_tokens.append([str(i)])

    vocab = build_vocab_from_iterator(all_tokens, specials=['[CLS]', '[SEP]', '[PAD]', '[EOS]', '[MASK]'])
    return vocab


# Lambda functions used recursively to faltten lists along the pre-processing 
flatten_list = lambda tensor, dimension:[spectra for sample in tensor for spectra in sample[dimension]]
flatten_sample = lambda sample,dimension:[spectra for spectra in sample[dimension]]


## MODEL USAGE FUNCTIONS for training, evaluation and plotting results

# BERT TRAINING

def BERT_train(model: nn.Module, optimizer, criterion, scheduler, dataset: list, results_file, batchsize: int, current_epoch: int, total_epochs: int, \
    limited_seq_len: int, shorter_len_perc: int, log_interval: int, device, scaler = None):
    """Training function for the training of BERT.

    Parameters:
    - model: nn.Module used to define the BERT model we want to train.
    - optimizer: optimizer used for training the model. Default: AdamW.
    - criterion: Criterion used for the backpropagation. Default: CrossEntropy.
    - scheduler: Learning rate scheduler to implement warmup and later decay as defined in BERT paper.
                 Default: class BERTscheduler defined in architectures.
    - dataset: Dataset used for the training.
    - results_file: Pathway to file where errors will be written.
    - batchsize: Batchsize defined for the training.
    - current_epoch: Number of epoch within training for sentence shortening purposes.
    - total_epochs: Number of total epochs for sentence shortening purposes.
    - limited_seq_len: Length of sentences during the training process we shorten the sentences.
                Comment: Process done 90% training in original BERT paper to accelerate training.
    - shorter_len_perc: Percentage training that we want the sentences to be shortened.
    - log_interval: batch interval we want to use to write loss status among other values.
    - device: Device where code is running. Either cpu or cuda
    - scaler: Scaler used to optimize training time by reducing from float32 to float16 where possible.
     """

    model.train()  # turn on train mode
    total_loss = 0.
    epoch_status = round(current_epoch/total_epochs, 2) # Get training stage to do sequence shortening if required
    start_time = time.time()

    inputs, targets, src_mask, labels, num_batches = get_training_batch(dataset, batchsize, epoch_status, limited_seq_len, shorter_len_perc)
    # Batching data after randomizing the order among all the scans of all the training samples with the defined batch size
    # QUESTION: Should the batching be done outside the train loop and use the same batches for all epochs?

    for batch in range(num_batches):
        with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.autocast(device_type=device.type): # optimize training time with scaler (float32 --> float16)
            output = model(inputs[batch].to(device), src_mask[batch].to(device))

            loss = 0
            for data_sample in range(batchsize): # each spectrum has different masking schema/masked positions --> !!PROBLEM: Although we define a batchsize, we end up calculating the loss 1 by 1
                single_output = output[:, data_sample] 
                single_target = targets[batch][:, data_sample]
                labels_output = labels[batch][data_sample]

                # Get just output and target in masked positions, which are the ones used for training our model
                mask_mapping = map(single_output.__getitem__, labels_output) 
                batch_output = torch.stack(list(mask_mapping)).to(device)

                target_mapping = map(single_target.__getitem__, labels_output)
                batch_target = torch.stack(list(target_mapping)).to(device)

                loss += criterion(batch_output, batch_target)    

        optimizer.zero_grad()
        scaler.scale(loss).backward() if scaler else loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.scale(optimizer) if scaler else optimizer.step()
        scaler.update() if scaler else None
        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_lr()
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            results_file.write(f'| epoch {current_epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} \n')     
            results_file.flush()
            total_loss = 0
            start_time = time.time()

        scheduler.step()

    return total_loss/num_batches


def BERT_evaluate(model: nn.Module, criterion, dataset: list, results_file, batchsize: int, current_epoch: int, total_epochs: float, \
    limited_seq_len: int, shorter_len_perc: int, start_time, device) -> float:
    """Evalutation function for the training of BERT.

    Parameters:
    - model: nn.Module used to define the BERT model we want to train.
    - criterion: Criterion used for the backpropagation. Default: CrossEntropy.
    - dataset: Dataset used for the training.
    - results_file: Pathway to file where errors will be written.
    - batchsize: Batchsize defined for the training.
    - current_epoch: Number of epoch within training for sentence shortening purposes.
    - total_epochs: Number of total epochs for sentence shortening purposes.
    - limited_seq_len: Length of sentences during the training process we shorten the sentences.
                Comment: Process done 90% training in original BERT paper to accelerate training.
    - shorter_len_perc: Percentage training that we want the sentences to be shortened.
    - start_time: Start time of the epoch for simple time control on the results_file.
    - device: Device where code is running. Either cpu or cuda
     """

    model.eval()  # turn on evaluation mode
    epoch_status = round(current_epoch/total_epochs, 2)
    total_loss = 0.
    
    inputs, targets, src_mask, labels, num_batches = get_training_batch(dataset, batchsize, epoch_status, limited_seq_len, shorter_len_perc)
    with torch.no_grad():
        for batch in range(num_batches):
            with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.autocast(device_type=device.type):
                output = model(inputs[batch].to(device), src_mask[batch].to(device))
                loss = 0
                for data_sample in range(batchsize):
                    single_output = output[:, data_sample]
                    single_target = targets[batch][:, data_sample]
                    labels_output = labels[batch][data_sample]

                    mask_mapping = map(single_output.__getitem__, labels_output)
                    batch_output = torch.stack(list(mask_mapping)).to(device)

                    target_mapping = map(single_target.__getitem__, labels_output)
                    batch_target = torch.stack(list(target_mapping)).to(device)

                    loss += criterion(batch_output, batch_target)
                
                total_loss += batchsize * loss.item()
    
    default_val_loss = total_loss / (batchsize - 1)
    val_loss = total_loss / (batchsize * num_batches)

    elapsed = time.time() - start_time

    results_file.write(f'| end of epoch {current_epoch:3d} | time: {elapsed:5.2f}s | '
          f'default validation loss {default_val_loss:5.2f}| validation loss {val_loss:5.2f} \n')
    results_file.flush()

    return val_loss


def get_training_batch(data_ds, batchsize, epoch_status, limited_seq_len, shorter_len_perc):
    # Create function to divide input, target, att_mask and labels in batches
    # Transpose by .t() method because we want the size [seq_len, batch_size]

    # This function returns a list with len num_batches = len(input_data) // batchsize
    # where each element of this list contains a tensor of size [seq_len, batch_size] or transposed
    # Thus, looping through the whole list with len num_batches, we are going through 
    # the whole dataset, but passing it to the model in tensors of batch_size.

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
            labels_batch.append([[val for val in labels[index] if val < limited_seq_len] for index in batch_indices])
        else:
            input_batch.append(torch.stack([input_data[index] for index in batch_indices]).t())
            target_batch.append(torch.stack([target_data[index] for index in batch_indices]).t())
            att_mask_batch.append(torch.stack([att_mask[index] for index in batch_indices]))
            labels_batch.append([labels[index] for index in batch_indices])

    return input_batch, target_batch, att_mask_batch, labels_batch, num_batches


def plot_training_error(epochs, training_error, validation_error, pathway):
    plt.plot(range(epochs), training_error, label='Training error')
    plt.plot(range(epochs), validation_error, label='Validation error')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(pathway)


# BERT FINE-TUNING

def BERT_finetune_train(BERT_model: nn.Module, finetune_model: nn.Module, optimizer, criterion, learning_rate: float, dataset: list, \
    results_file, batchsize: int, epoch: int, log_interval, device, same_sample: bool=True, scaler = None) -> None:

    finetune_model.train()
    total_loss = 0
    log_interval = 1
    start_time = time.time()
    att_weights_matrix = []
    for sample in range(len(dataset)):

        inputs, class_labels, num_batches = get_finetune_batch(dataset[sample], batchsize, same_sample)

        #num_class_spectra = top_attention_perc * len(inputs) TODO!! Just consider x% most relevant scans determined by attention

        inside_train_error = []
        hidden_vectors_sample = []
        with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.autocast(device_type=device.type):
            for batch in range(num_batches):  
                hidden_vectors_batch = BERT_model(inputs[batch].to(device))
                hidden_vectors_sample.append(hidden_vectors_batch)
            #print('BERT output size: ', hidden_vectors_sample[0].size())
            #print('Attention layer input size: ', torch.cat(hidden_vectors_sample).size())
            att_weights, output = finetune_model(torch.cat(hidden_vectors_sample).to(device))
            #print('Final output size: ', output.size())
            
            att_weights_matrix.append(att_weights)
        '''
        max_size = max([len(sample) for sample in att_weights_matrix])
        best_att_weights_matrix = [tensor.tolist() for tensor in att_weights_matrix]
   
        for sample in best_att_weights_matrix:
            if len(sample) < max_size:
                sample.append([0]*(max_size - len(sample)))

        plt.matshow(best_att_weights_matrix)
        plt.xlabel('Scans / Spectra')
        plt.ylabel('Samples / Patients')
        plt.colorbar()
        plt.show()
        '''
        '''
        # If at some point we wanna take the x% most informative spectra
        _, used_spectra = torch.topk(att_weights, num_class_spectra)

        used_output = output[used_spectra].to(device)
        used_labels = class_labels[used_spectra].to(device)
        '''
        loss = criterion(torch.squeeze(output), torch.cat(class_labels))
        optimizer.zero_grad()
        scaler.scale(loss).backward() if scaler else loss.backward()
        torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), 0.5)
        scaler.scale(optimizer) if scaler else optimizer.step()
        scaler.update() if scaler else None
        total_loss += loss.item()

        if sample % log_interval == 0 and sample > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            inside_train_error.append(cur_loss)
            #print(f'| epoch {epoch:3d} | {sample:2d}/{len(dataset):1d} samples | '
            #f'lr {learning_rate:02.5f} | ms/sample {ms_per_batch:5.2f} | '
            #f'loss {loss:5.2f} \n')
            results_file.write(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
            f'lr {learning_rate:02.2f} | ms/batch {ms_per_batch:5.2f} | '
            f'loss {loss:5.2f} \n')     
            results_file.flush()
            total_loss = 0
            start_time = time.time()

    return inside_train_error, att_weights_matrix


def BERT_finetune_evaluate(BERT_model: nn.Module, finetune_model: nn.Module, criterion, dataset: list, results_file, batchsize: int, \
    current_epoch: int, start_time, device) -> float:

    finetune_model.eval()
    total_loss = 0.
    # For validation, we don't just pass one sample at a time, but randomize
    inputs, class_labels, num_batches = get_finetune_batch(dataset, batchsize, same_sample = False)
    
    with torch.no_grad():
        for batch in range(num_batches):
            hidden_vectors = BERT_model(inputs[batch].to(device))
            _, output = finetune_model(hidden_vectors.to(device))
            loss = criterion(output, class_labels[batch].to(device))
            total_loss += loss.item()
        
    val_loss = total_loss / num_batches
    elapsed = time.time() - start_time

    results_file.write(f'| end of epoch {current_epoch:3d} | time: {elapsed:5.2f}s | '
          f'validation loss {val_loss:5.2f} \n')
    results_file.flush()

    return val_loss


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


def update_best_att_matrix(att_weights_matrix):
    # Save the attention weights for this best performing model
    max_size = max([len(sample) for sample in att_weights_matrix])
    best_att_weights_matrix = [tensor.tolist() for tensor in att_weights_matrix]
    for sample in best_att_weights_matrix:
        if len(sample) < max_size:
            sample.extend([[0]]*(max_size - len(sample)))
    
    return best_att_weights_matrix


## Plotting fucntions

def plot_BERT_training_error(epochs, training_error, validation_error, pathway):
    plt.plot(range(epochs), training_error, label='Training error')
    plt.plot(range(epochs), validation_error, label='Validation error')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(pathway)


def plot_finetuning_error(epochs, training_error, validation_error, pathway):
    plt.plot(range(epochs), training_error, label='Training error')
    plt.plot(range(epochs), validation_error, label='Validation error')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(pathway)


def plot_att_weights(att_weights_matrix, pathway):
    plt.matshow(att_weights_matrix)
    plt.xlabel('Scans / Spectra')
    plt.ylabel('Samples / Patients')
    plt.colorbar()
    plt.savefig(pathway)


def plot_embeddings(model: nn.Module, samples_names: list, dataset: list, \
             batchsize: int, aggregation_layer: string, device: torch.device):

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
    plt.savefig('/home/projects/cpr_10006/people/enrcop/Figures/Embeddings/BERT_vanilla_HPvsALD.png')
    #plt.savefig(os.getcwd() + '/Embedding.png')
    #plt.show()
      



        
