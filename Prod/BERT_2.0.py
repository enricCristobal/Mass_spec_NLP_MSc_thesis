#!/usr/bin/env python

import os
import time
import copy
import random
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from data_load import data_load
#from BERT_model import TransformerModel, train, evaluate

import math
from typing import Tuple
from torch.utils.data import dataset

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
output_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens/scan_desc_tokens/'
input_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/ms1_scans/'

evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_evolution.txt', "w")
evolution_file.write(str(device))
evolution_file.flush()

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
file_limit = int(len(samples_list) * training_percentage)
training_samples = samples_list[:file_limit]
#training_samples = samples_list[:5]
#val_samples = samples_list[5:7]
#testing_samples = samples_list[file_limit:file_limit+1] ## TODO!: Fix later to make the training, testing and validation partitioning more dynamic
val_samples = samples_list[file_limit:]

def get_batch(input_data, target_data, att_mask, batchsize):
    # Transpose by .t() method because we want the size [seq_len, batch_size]
    # This function returns a list with len num_batches = len(input_data) // batchsize
    # where each element of this list contains a tensor of size [seq_len, batch_size]
    # Thus, looping through the whole list with len num_batches, we are going through 
    # the whole dataset, but passing it to the model in tensors of batch_size.
    num_batches = len(input_data) // batchsize
    input_batch = []; target_batch = []; att_mask_batch = []
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
    return input_batch, target_batch, att_mask_batch, num_batches

# Create function to divide input, target, att_mask for each dataloader
flatten_list = lambda tensor, dimension:[spectra for sample in tensor for spectra in sample[dimension]]

# Create train dataloader 
# (all input_ds, target_ds and att_mask_ds will have size #spectra x 512 --> each spectra is an input to train our model )

train_ds = [data.create_dataset(f,vocab) for f in training_samples]
input_train_ds = flatten_list(train_ds, 0)
target_train_ds = flatten_list(train_ds, 1)
att_mask_train_ds = flatten_list(train_ds, 2)

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
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(output)
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
emsize = 768  # embedding dimension
d_hid = 768  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 12  # number of heads in nn.MultiheadAttention
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
lr = 1e-4  # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 50
    start_time = time.time()

    inputs, targets, src_mask, num_batches = get_batch(input_train_ds, target_train_ds, att_mask_train_ds, batchsize)
    
    for batch in range(num_batches):
        output = model(inputs[batch].to(device), src_mask[batch].to(device))
        # TODO!! Add some way to just calculate the lost on the masked inputs!
        loss = criterion(output.view(-1, ntokens), targets[batch].reshape(-1).to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            evolution_file.write(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f} \n')     
            evolution_file.flush()
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module) -> float:
    
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    inputs, targets, src_mask, num_batches = get_batch(input_val_ds, target_val_ds, att_mask_val_ds, batchsize)
    with torch.no_grad():
        for batch in range(num_batches):
            output = model(inputs[batch], src_mask[batch])
            output_flat = output.view(-1, ntokens)
            total_loss += batchsize * criterion(output_flat, targets[batch].reshape(-1)).item()
    return total_loss / (batchsize - 1)

best_val_loss = float('inf')
epochs = 50
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time

    evolution_file.write('-' * 89)
    evolution_file.write(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    evolution_file.write('-' * 89)
    evolution_file.flush()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model, 'bert_model_vanilla.pt')

    scheduler.step()

evolution_file.close()
'''
test_loss = evaluate(best_model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)
'''
