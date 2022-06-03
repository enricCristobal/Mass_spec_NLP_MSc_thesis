#!/usr/bin/env python

import string
import torch
from torch import nn, softmax
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
import copy

import pandas as pd
import random
import time
from statistics import mean
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.metrics as skl

from architectures import *

torch.manual_seed(52)
torch.cuda.manual_seed(52)

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
    print_loss = 0.
    total_loss = 0.
    epoch_status = round(current_epoch/total_epochs, 2) # Get training stage to do sequence shortening if required
    start_time = time.time()

    inputs, targets, padding_mask, masking_labels, num_batches = get_training_batch(dataset, batchsize, epoch_status, limited_seq_len, shorter_len_perc)
    # Batching data after randomizing the order among all the scans of all the training samples with the defined batch size
    #with torch.cuda.amp.GradScaler():
    for batch in range(num_batches):
        output = model(inputs[batch].to(device), padding_mask[batch].to(device))
        loss = 0
        for scan in range(batchsize): # each spectrum has different masking schema/masked positions
            single_output = output[:, scan] # predicted output by BERT and decoder
            single_target = targets[batch][:, scan] # expected target for this spectrum
            labels_output = masking_labels[batch][scan] # components of vector that were masked --> must be used for backpropagation
            # Get just output and target in masked positions, which are the ones that must be used for training our model
            batch_output = torch.stack(list(map(single_output.__getitem__, labels_output)))
            batch_output = batch_output.to(device)
            
            batch_target = torch.stack(list(map(single_target.__getitem__, labels_output)))
            batch_target = batch_target.to(device)

            loss += criterion(batch_output, batch_target)    

        optimizer.zero_grad()
        scaler.scale(loss).backward() if scaler else loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer) if scaler else optimizer.step()
        scaler.update() if scaler else None
        print_loss += loss.item() / batchsize
        total_loss += loss.item() / batchsize
        
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_lr()
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = print_loss / log_interval
            results_file.write(f'| epoch {current_epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:e} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} \n')     
            results_file.flush()
            print_loss = 0
            start_time = time.time()

        scheduler.step() # As it is done in BERT paper, we change learning rate at each batch/step
    
    return total_loss/ num_batches


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
    
    inputs, targets, padding_mask, masking_labels, num_batches = get_training_batch(dataset, batchsize, epoch_status, limited_seq_len, shorter_len_perc)
    #with torch.cuda.amp.GradScaler():
    with torch.no_grad():
        for batch in range(num_batches):
            output = model(inputs[batch].to(device), padding_mask[batch].to(device))
            loss = 0
            for scan in range(batchsize):
                single_output = output[:, scan]
                single_target = targets[batch][:, scan]
                labels_output = masking_labels[batch][scan]

                batch_output = torch.stack(list(map(single_output.__getitem__, labels_output))).to(device)
                batch_output = batch_output.to(device)

                batch_target = torch.stack(list(map(single_target.__getitem__, labels_output))).to(device)
                batch_target = batch_target.to(device)

                loss += criterion(batch_output, batch_target)
          
            total_loss += loss.item() / batchsize
    
    val_loss = total_loss /  num_batches

    elapsed = time.time() - start_time

    results_file.write(f'| end of epoch {current_epoch:3d} | time: {elapsed:5.2f}s | validation loss {val_loss:5.2f} \n')
    results_file.flush()

    return val_loss


def get_training_batch(data_ds, batchsize, epoch_status, limited_seq_len, shorter_len_perc):
    """Create function to divide input, target, att_mask and labels in batches
     Transpose by .t() method because we want the size [seq_len, batch_size]
    This function returns a list with len num_batches = len(input_data) // batchsize
    where each element of this list contains a tensor of size [seq_len, batch_size] or transposed
    Thus, looping through the whole list with len num_batches, we are going through 
    the whole dataset, but passing it to the model in tensors of batch_size."""

    input_data = flatten_list(data_ds, 0)
    target_data = flatten_list(data_ds, 1)
    padding_mask = flatten_list(data_ds, 2)
    masking_labels = flatten_list(data_ds, 3)

    num_batches = len(input_data) // batchsize

    input_batch = []; target_batch = []; att_mask_batch = []; labels_batch = []
    indices = list(range(len(input_data)))
    
    for shuffling_times in range(10):
        random.shuffle(indices)

    for batch in range(num_batches):
        batch_indices = []
        for _ in range(batchsize): # we want all batches to be same size, avoid the last batch if there are some fewer datapoints left
            batch_indices.append(indices.pop())
  
        if epoch_status <= shorter_len_perc:
            input_batch.append(torch.stack([input_data[index][:limited_seq_len] for index in batch_indices]).t())
            target_batch.append(torch.stack([target_data[index][:limited_seq_len] for index in batch_indices]).t())
            att_mask_batch.append(torch.stack([padding_mask[index][:limited_seq_len] for index in batch_indices])) # because size passed needs to be (N, S), being N batchsize and S seq_len
            # Just get those smaller than limited_seq_len          
            labels_batch.append([[val for val in masking_labels[index] if val < limited_seq_len] for index in batch_indices])
            
        else:
            input_batch.append(torch.stack([input_data[index] for index in batch_indices]).t())
            target_batch.append(torch.stack([target_data[index] for index in batch_indices]).t())
            att_mask_batch.append(torch.stack([padding_mask[index] for index in batch_indices]))
            labels_batch.append([masking_labels[index] for index in batch_indices])

    return input_batch, target_batch, att_mask_batch, labels_batch, num_batches


def plot_training_error(epochs, training_error, validation_error, pathway):
    plt.plot(range(1, epochs+1), training_error, label='Training error')
    plt.plot(range(1, epochs+1), validation_error, label='Validation error')
    plt.title('Training and validation error BERT model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(pathway)


# BERT FINE-TUNING

def define_architecture(class_layer: str, att_matrix: bool, n_input: int, num_labels: int, n_layers_attention: int, n_units_attention: int, \
    num_channels: int, kernel_size: int, padding: int, n_units_linear_CNN: int, n_layers_linear: int, n_units_linear: int):

    if class_layer == 'CNN':
        #scans_count = min_scan_count // batchsize * batchsize # this is the number of scans that will be obtained after the BERT model to pass to CNN
        attention_network = AttentionNetwork(n_input_features = n_input, n_layers = n_layers_attention, n_units = n_units_attention)
        classification_layer = CNNClassificationLayer(num_labels=num_labels,num_channels=num_channels, kernel = kernel_size, padding = padding, n_units=n_units_linear_CNN)
        model = FineTune_classification(attention_network, classification_layer)
        BERT_fine_tune_train = BERT_finetune_train_att_matrix

    else:
        if att_matrix:
            attention_network = AttentionNetwork(n_input_features = n_input, n_layers = n_layers_attention, n_units = n_units_attention)
            classification_layer = LinearClassificationLayer(n_input_features=n_input, num_labels = num_labels, n_layers=n_layers_linear, n_units=n_units_linear)
            model = FineTune_classification(attention_network, classification_layer)
            BERT_fine_tune_train = BERT_finetune_train_att_matrix

        else:
            model = LinearClassificationLayer(n_input_features=n_input, num_labels=num_labels, n_layers=n_layers_linear, n_units=n_units_linear)
            BERT_fine_tune_train = BERT_finetune_train
    
    return model, BERT_fine_tune_train


def BERT_finetune_train(BERT_model: nn.Module, finetune_model: nn.Module, optimizer, criterion, learning_rate: float, dataset: list, \
    results_file, batchsize: int, epoch: int, write_interval: int, device, scaler = None) -> None:

    BERT_model.eval()
    finetune_model.train()
    print_loss = 0.
    total_loss = 0.
    start_time = time.time()

    inputs, class_labels = get_finetune_batch(dataset, batchsize, same_sample=False) 
    
    for batch in range(len(inputs)): 
        with torch.no_grad():
            hidden_vectors_batch = BERT_model(inputs[batch].to(device))
            hidden_vectors_batch_cpu = hidden_vectors_batch.cpu()
            del hidden_vectors_batch
            
        output = finetune_model(hidden_vectors_batch_cpu)
        loss = criterion(output, class_labels[batch])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), 0.5)
        scaler.step(optimizer) if scaler else optimizer.step()
        scaler.update() if scaler else None
        print_loss += loss.item()
        total_loss += loss.item()
       
        if batch % write_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / write_interval
            cur_loss = print_loss / write_interval
            results_file.write(f'| epoch {epoch:3d} | {batch:5d}/{len(inputs):5d} batches | '
            f'lr {learning_rate:e} | ms/batch {ms_per_batch:5.2f} | '
            f'loss {cur_loss:5.2f} \n')     
            results_file.flush()
            print_loss = 0
            start_time = time.time()

    return total_loss/len(inputs), None


def BERT_finetune_train_att_matrix(BERT_model: nn.Module, finetune_model: nn.Module, optimizer, criterion, learning_rate: float, dataset: list, \
    results_file, batchsize: int, epoch: int, write_interval: int, device, scaler = None) -> None: # top_attention_perc: float = None,

    BERT_model.eval()
    finetune_model.train()
    print_loss = 0.
    total_loss = 0.
    start_time = time.time()
    att_weights_matrix = []

    for sample in range(len(dataset)):
        inputs, class_labels = get_finetune_batch(dataset[sample], batchsize, same_sample=True) 

        #if top_attention_perc:
        #    num_class_spectra = top_attention_perc * len(inputs) #TODO!! Just consider x% most relevant scans determined by attention

        hidden_vectors_sample = []
        with torch.no_grad():
            for batch in range(len(inputs)): 
                hidden_vectors_batch = BERT_model(inputs[batch].to(device))
                hidden_vectors_batch_cpu = hidden_vectors_batch.cpu()
                del hidden_vectors_batch
                hidden_vectors_sample.append(hidden_vectors_batch_cpu) 
        
        att_weights, output = finetune_model(torch.cat(hidden_vectors_sample))
        att_weights_matrix.append(att_weights)

        '''
        # If at some point we wanna take the x% most informative spectra
        _, used_spectra = torch.topk(att_weights, num_class_spectra)
        used_output = output[used_spectra].to(device)
        used_labels = class_labels[used_spectra].to(device)
        '''

        loss = criterion(output, torch.cat(class_labels)[:len(output)])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), 0.5)
        scaler.step(optimizer) if scaler else optimizer.step()
        scaler.update() if scaler else None
        print_loss += loss.item()
        total_loss += loss.item()

        if sample % write_interval == 0 and sample > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / write_interval
            cur_loss = print_loss / write_interval
            results_file.write(f'| epoch {epoch:3d} | {batch:5d}/{len(inputs):5d} batches | '
            f'lr {learning_rate:e} | ms/batch {ms_per_batch:5.2f} | '
            f'loss {cur_loss:5.2f} \n')     
            results_file.flush()
            print_loss = 0
            start_time = time.time()

    return total_loss/len(dataset), att_weights_matrix


def BERT_finetune_evaluate(BERT_model: nn.Module, finetune_model: nn.Module, att_matrix: bool, class_layer: str, criterion, dataset: list, results_file, batchsize: int, \
    current_epoch: int, start_time, device) -> float:

    BERT_model.eval()
    finetune_model.eval()
    total_loss = 0.
    #n_correct_predictions = 0
    y_true = []; y_pred = []
    if att_matrix: 
        for sample in range(len(dataset)):

            inputs, class_labels = get_finetune_batch(dataset[sample], batchsize, same_sample = True)
            hidden_vectors_sample = []
            with torch.no_grad():
                for batch in range(len(inputs)): 
                    hidden_vectors = BERT_model(inputs[batch].to(device))
                    hidden_vectors_cpu = hidden_vectors.cpu()
                    del hidden_vectors
                    hidden_vectors_sample.append(hidden_vectors_cpu) 

            _, output = finetune_model(torch.cat(hidden_vectors_sample))
            loss = criterion(output,torch.cat(class_labels)[:len(output)])
            total_loss += loss.item()
            softmax = nn.Softmax(dim=1)
            _, predicted = torch.max(softmax(output), 1)
            y_pred.append(predicted.numpy())
            
            if class_layer == 'CNN':
                y_true.append(class_labels[0][0].tolist())
                val_loss = total_loss / len(dataset)
            else:
                y_true.append(torch.cat(class_labels).tolist())   
                val_loss = total_loss / (len(output)*len(dataset))
                          
    else:
        # For validation, we don't just pass one sample at a time, but randomize
        inputs, class_labels = get_finetune_batch(dataset, batchsize, same_sample = False)
        
        for batch in range(len(inputs)): # len(inputs) == num_batches    
            with torch.no_grad():
                hidden_vectors_batch = BERT_model(inputs[batch].to(device))
                hidden_vectors_batch_cpu = hidden_vectors_batch.cpu()
                del hidden_vectors_batch

            output = finetune_model(hidden_vectors_batch_cpu)
            loss = criterion(output, class_labels[batch])
            total_loss += loss.item()
            softmax = nn.Softmax(dim=1)
            _, predicted = torch.max(softmax(output), 1)

            y_pred.append(predicted.numpy())
            y_true.append(class_labels[batch].tolist())

        val_loss = total_loss / len(inputs)

    y_pred = [item for sublist in y_pred for item in sublist]
    if class_layer != 'CNN':
        y_true = [item for sublist in y_true for item in sublist]

    acc = skl.accuracy_score(y_pred, y_true)
    
    elapsed = time.time() - start_time

    results_file.write(f'| end of epoch {current_epoch:3d} | time: {elapsed:5.2f}s | '
          f'validation loss {val_loss:5.2f} \n')
    results_file.flush()

    return val_loss, acc, y_true, y_pred


def get_metrics(y_true, y_pred, cm_fig_pathway, ROC_curve: bool, ROC_fig_pathway):
    
    cm = skl.confusion_matrix(y_true, y_pred)
    
    skl.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title('Confusion matrix')
    plt.savefig(cm_fig_pathway)

    TP = cm[0,0]; FN = cm[0,1]; FP = cm[1,0]; TN = cm[1,1]

    precision = TP/(FP + TP)
    recall = TP/(TP + FN)
    F1_score = 2*precision*recall/(precision+recall)
    
    if ROC_curve:
        fpr, tpr, _ = skl.roc_curve(y_true,  y_pred)
        auc = skl.roc_auc_score(y_true, y_pred)

        # create ROC curve
        plt.plot(fpr,tpr,label="AUC="+str(round(auc, 3)))
        plt.title('ROC curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.savefig(ROC_fig_pathway)

    return cm, precision, recall, F1_score


def get_finetune_batch(sample_data_ds, batchsize, same_sample: bool): 
    """Used for fine-tuning when both mixing spectra from different samples under the same batch and not (same_sample boolean)
    We consider the input is already just one sample.
    Parameters:
    - same_sample: we use same_sample as True when training because we want to get the attention matrix
    """

    flatten = flatten_sample if same_sample == True else flatten_list
    input_data = flatten(sample_data_ds, 0)
    labels_data = flatten(sample_data_ds, 1)
    
    num_batches = len(input_data) // batchsize
    
    # We do same process of randomizing a bit, so we don't follow the retention times of the experiment in order
    input_batch = []; labels_batch = []; 
    indices = list(range(len(input_data)))

    if not same_sample:
        for shuffling_times in range(10):
            random.shuffle(indices)

    for batch in range(num_batches):
        batch_indices = []
        for _ in range(batchsize):
            batch_indices.append(indices.pop())

        input_batch.append(torch.stack([input_data[index] for index in batch_indices]).t())
        labels_batch.append(torch.stack([labels_data[index] for index in batch_indices]))
    
    return input_batch, labels_batch


def update_best_att_matrix(att_weights_matrix):
    """Save the attention weights for this best performing model"""
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
    plt.title('Error evolution training process')
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


def plot_individual_axis_distribution(embedding, patient_type: str, axis: str):
    
    dim = 0 if axis == 'X' else 1
    mean = np.mean(embedding[:,dim])
    sd = np.std(embedding[:,dim])
    x = np.linspace(mean - 3*sd, mean + 3*sd, 100)
    plt.hist(embedding[:,dim], bins = 50, density = True)
    #plt.plot(x, stats.norm.pdf(x, mean, sd), linewidth=3)
    #plt.axvline(mean, color='red', linewidth = 1)
    plt.title('%s-%s axis embedding distribution (mean=%2.2f, std≈%2.2f)' %(patient_type, axis, mean, sd))
    plt.xlabel('%s-axis embedding' %(axis))
    plt.savefig('C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Computerome_results\\BERT_vanilla_embeddings\\Individual_analysis\\%s_%s_distribution.png' %(patient_type, axis))


def plot_BERT_embeddings(model: nn.Module, samples_names: list, dataset: list, \
             batchsize: int, aggregation_layer: string, labels, pathway: str, patient_type: str, device: torch.device):

    #assert aggregation_layer in ["sample", "ret_time"], "aggregation_layer must be either at 'sample' level or 'ret_time' level"

    model.eval()
    reducer = umap.UMAP(n_neighbors=15)
    dataframe = []
    
    for sample in range(len(dataset)):
        #print('Sample number: ', sample + 1)
        inputs, class_labels = get_finetune_batch(dataset[sample], batchsize, same_sample=True)

        hidden_vectors_sample = []
        for batch in range(len(inputs)):
            hidden_vectors_batch = model(inputs[batch].to(device)) # torch.Tensor of size [batchsize x embed_size]
            hidden_vectors_sample.append(hidden_vectors_batch.tolist()) 
        
        hidden_vectors = [scan for batched_scans in hidden_vectors_sample for scan in batched_scans] # after appending all batches --> list of size [# scans/sample x embed_size]

        # As preparation for the content "skeleton" of the dataframe, name of sample for each of its scans, the number of scan ("ret_time"), and the label for each scan of the sample again (class_labels[0][0], because all will have the same for each sample)
        samples_and_time = list(map(list,zip([samples_names[sample]]*len(dataset[sample][0]), range(len(dataset[sample][0])), [class_labels[0][0].tolist()]*len(dataset[sample][0]))))
        dataframe.append(list(map(list.__add__, samples_and_time, hidden_vectors))) # include embedding values for each scan of current sample
        
    flat_dataframe = [sample_scan for sample_dataframe in dataframe for sample_scan in sample_dataframe]
    df = pd.DataFrame(flat_dataframe, columns = ['Sample_name', 'Scan_order', 'Label'] + list(range(hidden_vectors_batch.size()[1]))) # creation of df colums with embed_size
    
    # Size df is [# total scans over all samples x (3 + embed_size)]
    if aggregation_layer == "sample":
        ## !!! Applying the mean here!!
        df = df.groupby('Sample_name').mean() # apply mean over each components for all embeddings part of the same sample
        # df size now is [#samples x (2 + embed_size)] (Sample_name has become the row name --> doesn't count as column)
        embedding = reducer.fit_transform(df.iloc[:,2:])
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df.Label)
        plt.legend(handles=scatter.legend_elements()[0], labels = labels, title="Patients status")
        plt.title('UMAP projection of the embedding space by patients', fontsize=14)
        plt.savefig(pathway)
        ## !! QUESTION: Is this the proper way to approach it? Felix got good clustering applying mean, but mathematical meaning?
    else:
        embedding = reducer.fit_transform(df.iloc[:,3:]) # Remove the sample_name, scan_order, and label of the fitting
        plot_individual_axis_distribution(embedding, patient_type, axis='X')
        plt.clf()
        plot_individual_axis_distribution(embedding, patient_type, axis='Y')
        plt.clf()

        for scan in range(len(embedding)):
            plt.plot(embedding[scan, 0], embedding[scan, 1], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green", alpha=1/len(embedding)*(scan+1))
        plt.title('Projection of embedding space by scans for %s patient' %(patient_type), fontsize=14)
        plt.xlabel('Projected embedding dim 1', fontsize=8)
        plt.ylabel('Projected embedding dim 2', fontsize=8)
        plt.savefig(pathway + '%s_patient_time_embedding.png' %(patient_type))


#BASELINE MODELS
##VAE

def final_loss_VAE(bce_loss, mu, log_var):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 -sigma^2)
    bce_loss: reconstruction loss
    mu: mean from the latent vector
    log_var: log varaiance from the latent vector
    """

    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE+KLD


def train_VAE(model, vocab, dataset, device, optimizer, criterion, batchsize):
    model.train()
    total_loss = 0.0
    batch_input = []
    labels = []
    latent_space = []
    final_len = len(dataset) // batchsize * batchsize
    for sample in range(len(dataset)):
        batch_input.append(torch.stack(flatten_sample(dataset[sample], 0)))
        labels.append(flatten_sample(dataset[sample], 1)[0].tolist())

        if len(batch_input) == batchsize:
            batch = torch.stack(batch_input)[:, None, :, :]/float(len(vocab)) # prepare size for CNN of VAE
            batch= batch.to(device)
            optimizer.zero_grad()
            latent_space_samples, reconstruction, mu, log_var = model(batch)
            latent_space.append(latent_space_samples.tolist())
            bce_loss = criterion(reconstruction, batch)
            loss = final_loss_VAE(bce_loss, mu, log_var)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            batch_input = []        
    train_loss = total_loss/final_len
    return train_loss, latent_space, labels[:final_len]


def val_VAE(model, vocab, dataset, device, optimizer, criterion, batchsize):
    model.eval()
    total_loss = 0.0
    batch_input = []
    final_len = len(dataset) // batchsize * batchsize

    for sample in range(len(dataset)):
        batch_input.append(torch.stack(flatten_sample(dataset[sample], 0)))

        if len(batch_input) == batchsize:
            batch = torch.stack(batch_input)[:, None, :, :]/float(len(vocab)) # prepare size for CNN of VAE
            batch= batch.to(device)
            optimizer.zero_grad()
            _, reconstruction, mu, log_var = model(batch)
            bce_loss = criterion(reconstruction, batch)
            loss = final_loss_VAE(bce_loss, mu, log_var)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            batch_input = []
    val_loss = total_loss/final_len
    return val_loss


def plot_VAE_embedding(latent_space_samples, labels, labels_names, save_dir):
    dim1_vec = []; dim2_vec = []
    latent_space = [item for sublist in latent_space_samples for item in sublist]
    for dim1, dim2 in latent_space:
        dim1_vec.append(dim1)
        dim2_vec.append(dim2)
    
    scatter = plt.scatter(dim1_vec,dim2_vec, c=labels)
    plt.title('VAE latent space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(handles=scatter.legend_elements()[0], labels = labels_names, title="Patients status")
    plt.savefig(save_dir)
    #plt.show()


def train_class_VAE(VAE_model, class_network, vocab, dataset, device, optimizer, criterion, batchsize):
    VAE_model.eval()
    class_network.train()
    total_loss = 0.0
    batch_input = []; labels = []
    final_len = len(dataset) // batchsize * batchsize
    
    for sample in range(len(dataset)):
        
        batch_input.append(torch.stack(flatten_sample(dataset[sample], 0)))
        labels.append(flatten_sample(dataset[sample], 1)[0])
        
        if len(batch_input) == batchsize:
            batch = torch.stack(batch_input)[:, None, :, :]/float(len(vocab)) # prepare size for CNN of VAE
            batch= batch.to(device)
            labels = torch.stack(labels)
            with torch.no_grad():
                latent_space_samples = VAE_model(batch) # no need to have gradient values for VAE
            latent_space_samples = latent_space_samples.cpu()
            output = class_network(latent_space_samples)
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            batch_input = []; labels = []       
    
    train_loss = total_loss/final_len

    return train_loss


def val_class_VAE(VAE_model, class_network, vocab, dataset, device, optimizer, criterion, batchsize):
    VAE_model.eval()
    class_network.eval()
    total_loss = 0.0
    batch_input = []; labels = []; y_true = []; y_pred = []
    final_len = len(dataset) // batchsize * batchsize
    softmax = nn.Softmax(dim=1)

    for sample in range(len(dataset)):
        batch_input.append(torch.stack(flatten_sample(dataset[sample], 0)))
        labels.append(flatten_sample(dataset[sample], 1)[0])

        if len(batch_input) == batchsize:
            batch = torch.stack(batch_input)[:, None, :, :]/float(len(vocab)) # prepare size for CNN of VAE
            batch= batch.to(device)
            labels = torch.stack(labels)
            with torch.no_grad():
                latent_space_samples = VAE_model(batch) # no need to have gradient values for VAE
            latent_space_samples = latent_space_samples.cpu()
            output = class_network(latent_space_samples)
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
            _, predicted = torch.max(softmax(output), 1)

            y_pred.append(predicted.numpy())
            y_true.append(labels.tolist())
            batch_input = []; labels = []
      
    y_pred = [item for sublist in y_pred for item in sublist]
    y_true = [item for sublist in y_true for item in sublist]
    
    acc = skl.accuracy_score(y_pred, y_true)

    val_loss = total_loss/final_len
    return val_loss, acc, y_true, y_pred
