#!/usr/bin/env python

from cgi import print_arguments
import os
import time
import copy
import random
import numpy as np
from statistics import mean

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from data_load_test import data_load
#from BERT_model import TransformerModel, train, evaluate

import math
from typing import Tuple
from torch.utils.data import dataset
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved
t = torch.cuda.get_device_properties(0).total_memory
print('Initial: ', r, a, f, t)
'''
# set vocabulary
# create list of all words as lists
all_tokens = []
num_bins = 50000

for i in range(num_bins):
    all_tokens.append([str(i)])

vocab = build_vocab_from_iterator(all_tokens, specials=['[CLS]', '[SEP]', '[PAD]', '[EOS]', '[MASK]'])

# Get Data specifying the directory where raw data is or we want to store
output_dir = os.getcwd() + '\\test_data\\' # if we store and want to store data in same directory as script
input_dir = os.getcwd() + '\\pickle_data\\'


evolution_file = open(os.getcwd() + '\\loss_evolution.txt', "w")
data = data_load(output_dir, num_bins) # we save files creating folder in script's directory

# If pickle files need to be converted into raw data text files
pickle_files = False # we can add some sys.arg when calling this code to specify if we want this step to 
#happen or raw data is already in place

if pickle_files:
    data.pickle_to_raw(input_dir, descending_order=True)

# Define parameters for dataloaders' creation
training_percentage = 0.8
batchsize = 10

# Get all samples' files divided in training and testing
samples_list = os.listdir(output_dir)
training_samples = samples_list[0:2]
testing_samples = samples_list[2:4]
val_samples = samples_list[5:7]

def get_batch(input_data, target_data, att_mask, labels, batchsize):
    # Transpose by .t() method because we want the size [seq_len, batch_size]
    # This function returns a list with len num_batches = len(input_data) // batchsize
    # where each element of this list contains a tensor of size [seq_len, batch_size]
    # Thus, looping through the whole list with len num_batches, we are going through 
    # the whole dataset, but passing it to the model in tensors of batch_size.
    #num_batches = len(input_data) // batchsize
    num_batches = 5
    input_batch = []; target_batch = []; att_mask_batch = []; labels_batch = []
    indices = list(range(len(input_data)))

    for shuffling_times in range(10):
        random.shuffle(indices)

    for batch in range(num_batches):
        #batch = random.sample(range(0,len(input_data)), batchsize)
        batch_indices = []

        for j in range(batchsize):
            batch_indices.append(indices.pop())

        input_batch.append(torch.stack([input_data[index] for index in batch_indices]).t())
        target_batch.append(torch.stack([target_data[index] for index in batch_indices]).t())
        att_mask_batch.append(torch.stack([att_mask[index] for index in batch_indices])) # because size passed needs to be (N, S), being N batchsize and S seq_len
        labels_batch.append([labels[index] for index in batch_indices])

    return input_batch, target_batch, att_mask_batch, labels_batch, num_batches

# Create function to divide input, target, att_mask for each dataloader
flatten_list = lambda tensor, dimension:[spectra for sample in tensor for spectra in sample[dimension]]

# Create train dataloader 
# (all input_ds, target_ds and att_mask_ds will have size #spectra x 512 --> each spectra is an input to train our model )

train_ds = [data.create_dataset(f,vocab) for f in training_samples]
input_train_ds = flatten_list(train_ds, 0)
target_train_ds = flatten_list(train_ds, 1)
att_mask_train_ds = flatten_list(train_ds, 2)
labels_train_ds = flatten_list(train_ds, 3)

# Create test dataloader
#input_test_ds = flatten_list([data.create_dataset(f, vocab)[0] for f in testing_samples])
#target_test_ds = flatten_list([data.create_dataset(f, vocab)[1] for f in testing_samples])
#test_att_mask = flatten_list([data.create_dataset(f, vocab)[2] for f in testing_samples])
#test_input_batch, test_target_batch, test_att_mask_batch  = get_batch(input_test_ds, target_test_ds, test_att_mask, batchsize)

# Create validation dataloader

val_ds = [data.create_dataset(f,vocab) for f in val_samples]
input_val_ds = flatten_list(val_ds, 0)
target_val_ds = flatten_list(val_ds, 1)
att_mask_val_ds = flatten_list(val_ds, 2)
labels_val_ds = flatten_list(val_ds, 3)


# Using torch.nn.modules.transformer library

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)#, activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_key_padding_mask: Tensor, shape [batch_size, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        #print('Input', src.size())
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        #print('Encoded', src.size())
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        #print('Pre output', output.size())
        output = self.decoder(output)
        #print('Output', output.size())
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Define needed things for running the model

# Define parameters for the Transformer to replicate the BERT model architecture

ntokens = len(vocab)  # size of vocabulary
emsize = 10 #768  # embedding dimension
d_hid = 10 #768  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 #12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 #12  # number of heads in nn.MultiheadAttention
dropout = 0.1  # dropout probability
#activation = torch.nn.GELU # activation function on the intermediate layer (For now with gelu is not working)
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
'''
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved
t = torch.cuda.get_device_properties(0).total_memory
print('Model: ', r,a,f, t)
print(torch.cuda.memory_summary())
'''
criterion = nn.CrossEntropyLoss()
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for tensor sizes
# passed to the crossnetropy loss. If we want to pas the input for our labeled input --> having a 
# random size of masked labels, the input will be [label_size x 50005], and then target must be
# the target values obtained for these random components with size [label_size], i.e. label_size = 64
# meaning there are 64 "words" masked
lr = 1e-4  # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

training_error = []; validation_error = []

def train(model: nn.Module) -> None:
    print('Training!')
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 10
    start_time = time.time()

    inputs, targets, src_mask, labels, num_batches = get_batch(input_train_ds, target_train_ds, att_mask_train_ds, labels_train_ds, batchsize)
    # inputs, targets and src_mask have a len of num_batches, where each component is a tensor of shape [seq_len x batchsize], and labels has 
    # a len of num_batches as well, but each component has len batch_size, and each of these containing the label_size, being the n labels that 
    # were masked for each mass_spec graph --> shape labels [num_batches x batchsize x label_size]
    
    #print(len(labels), len(labels[0]), len(labels[0][1]))
    #print(len(inputs), inputs[0].size())
    inside_train_error = []
    for batch in range(num_batches):
        #print('batch!')
        output = model(inputs[batch].to(device), src_mask[batch].to(device)).to(device)
        # Output has size [seq_len, batchsize, len(vocab)] --> [512, batchsize, 50005]
        # The way loss was calculated before for all values at a time, is not possible when
        # considering the labels, as these will have different sizes, and therefore, we can't
        # create a "square" matrix/tensor -->
        # previous solution: loss = criterion(output.view(-1, ntokens), targets[batch].reshape(-1).to(device))
        # --> solution is to go one by one and adding to the loss per batch
        # --> We get the specific positions/labels for each data_sample within the batch, meaning
        # for eah column of the output, and calculate the loss just based on the difference 
        # between those with size [label_size x 50005], where label_size will constantly change
        # as this is obtained randomly in mask_input() function in data_load.

        #print(targets[batch].size(), targets[batch].reshape(-1).size())
        loss = 0
        for data_sample in range(batchsize):
            single_output = output[:, data_sample] 
            single_target = targets[batch][:, data_sample]
            labels_output = labels[batch][data_sample]
            #print(len(single_output), single_output.size(), len(labels_output))
            mask_mapping = map(single_output.__getitem__, labels_output)
            batch_output = torch.stack(list(mask_mapping))
            # batch_output has size [label_size x 50005] containing just those columns that where masked
            # therefore, we need the target to be the "real words" in the target for those "words" that have bee
            # masked out in the input --> targets[batch][data_sample][labels_output]
            target_mapping = map(single_target.__getitem__, labels_output)
            batch_target = torch.stack(list(target_mapping))

            loss += criterion(batch_output, batch_target)    
        #print(batch_output[0][batch_target[0]], batch_target[0])
        #print(output.view(-1, ntokens).size())
        #loss = criterion(output.view(-1, ntokens), targets[batch].reshape(-1).to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            inside_train_error.append(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} \n')     
            #evolution_file.flush()
            total_loss = 0
            start_time = time.time()
    training_error.append(mean(inside_train_error))

def evaluate(model: nn.Module) -> float:
    print('Evaluating!')
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    inputs, targets, src_mask, labels, num_batches = get_batch(input_val_ds, target_val_ds, att_mask_val_ds, batchsize)
    with torch.no_grad():
        for batch in range(num_batches):
            output = model(inputs[batch], src_mask[batch])
            loss = 0
            for data_sample in range(batchsize):
                single_output = output[:, data_sample]
                single_target = targets[batch][:, data_sample]
                labels_output = labels[batch][data_sample]

                mask_mapping = map(single_output.__getitem__, labels_output)
                batch_output = torch.stack(list(mask_mapping))

                target_mapping = map(single_target.__getitem__, labels_output)
                batch_target = torch.stack(list(target_mapping))

                loss += criterion(batch_output, batch_target)
            
            total_loss += batchsize * loss.item()
    validation_error.append(total_loss / (batchsize - 1))
    return total_loss / (batchsize - 1)

best_val_loss = float('inf')
epochs = 5
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time

    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print('Better model!')
        #torch.save(model, 'bert_model_vanilla.pt')

    scheduler.step()

plt.plot(range(epochs), training_error, range(epochs), validation_error)
plt.show()

evolution_file.close()
'''
test_loss = evaluate(best_model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)
'''
