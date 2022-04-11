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
output_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens/scan_desc_tokens_CLS/'
input_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/ms1_scans/'

evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/train_loss_cls_ret_halfsize.txt', "w")
#evolution_file.write(str(device))
#evolution_file.flush()

data = data_load(output_dir, num_bins) # we save files creating folder in script's directory

# If pickle files need to be converted into raw data text files
pickle_files = False # we can add some sys.arg when calling this code to specify if we want this step to 
#happen or raw data is already in place

if pickle_files:
    data.pickle_to_raw(input_dir, descending_order=True)

# Define parameters for dataloaders' creation
training_percentage = 0.7
val_percentage = 0.2
batchsize = 24

# Get all samples' files divided in training and testing
samples_list = os.listdir(output_dir)
file_limit_train = int(len(samples_list) * training_percentage)
file_limit_val = int(len(samples_list) * val_percentage)
training_samples = samples_list[:file_limit_train]
val_samples = samples_list[file_limit_train:(file_limit_train + file_limit_val)]

# Create function to divide input, target, att_mask for each dataloader
flatten_list = lambda tensor, dimension:[spectra for sample in tensor for spectra in sample[dimension]]

def get_batch(data_ds, batchsize, epoch_status, limited_seq_len):
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
        #batch = random.sample(range(0,len(input_data)), batchsize)
        batch_indices = []

        for j in range(batchsize):
            batch_indices.append(indices.pop())

        if epoch_status <= 0.9:
            input_batch.append(torch.stack([input_data[index][:limited_seq_len] for index in batch_indices]).t())
            target_batch.append(torch.stack([target_data[index][:limited_seq_len] for index in batch_indices]).t())
            att_mask_batch.append(torch.stack([att_mask[index][:limited_seq_len] for index in batch_indices])) # because size passed needs to be (N, S), being N batchsize and S seq_len
            # Just get those smaller than limited_seq_len
            smaller_labels = np.array([labels[index] for index in batch_indices])
            labels_batch.append(smaller_labels[smaller_labels<=limited_seq_len].tolist())
            print('Smaller', len(input_batch[-1]))
            print('Smaller labels:', labels_batch[-1])
        else:
            input_batch.append(torch.stack([input_data[index] for index in batch_indices]).t())
            target_batch.append(torch.stack([target_data[index] for index in batch_indices]).t())
            att_mask_batch.append(torch.stack([att_mask[index] for index in batch_indices]))
            labels_batch.append([labels[index] for index in batch_indices])

    return input_batch, target_batch, att_mask_batch, labels_batch, num_batches

# Create dataloaders considering the presence or not presence of CLS tokens for the masking, as well as the limited seq_length for 90% of training as mentioned in BERT paper

cls_token = True
limited_seq_len = 128
perc_epochs_shorter_seq_len = 0.9

# Create train dataloader 
# (all input_ds, target_ds and att_mask_ds will have size #spectra x 512 --> each spectra is an input to train our model )
train_ds = [data.create_training_dataset(f,vocab, cls_token) for f in training_samples]

# Create validation dataloader
val_ds = [data.create_training_dataset(f,vocab, cls_token) for f in val_samples]

# Using torch.nn.modules.transformer library

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, activation, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation)
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
emsize = 384 #192 # embedding dimension
d_hid = 384  #192 # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6  # 3 # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 6  # 3 # number of heads in nn.MultiheadAttention
dropout = 0.1  # dropout probability
activation = F.gelu # activation function on the intermediate layer (For now with gelu is not working)
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, activation, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 1e-4  # learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters = 10000)

scaler = torch.cuda.amp.GradScaler()

training_error = []; validation_error = []

def train(model: nn.Module) -> None:

    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 1000
    start_time = time.time()

    inputs, targets, src_mask, labels, num_batches = get_batch(train_ds, batchsize, epoch_status, limited_seq_len)
    
    inside_train_error = []

    for batch in range(num_batches):
        with torch.cuda.amp.autocast():
            output = model(inputs[batch].to(device), src_mask[batch].to(device))

            loss = 0
            for data_sample in range(batchsize): # each spectrum has different masking schema/masked positions
                single_output = output[:, data_sample] 
                single_target = targets[batch][:, data_sample]
                labels_output = labels[batch][data_sample]

                mask_mapping = map(single_output.__getitem__, labels_output)
                batch_output = torch.stack(list(mask_mapping)).to(device)

                target_mapping = map(single_target.__getitem__, labels_output)
                batch_target = torch.stack(list(target_mapping)).to(device)

                loss += criterion(batch_output, batch_target)    

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.scale(optimizer)
        #optimizer.step()
        scaler.update()
        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            inside_train_error.append(cur_loss)
            evolution_file.write(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} \n')     
            evolution_file.flush()
            total_loss = 0
            start_time = time.time()
    training_error.append(mean(inside_train_error))

def evaluate(model: nn.Module) -> float:

    model.eval()  # turn on evaluation mode
    total_loss = 0.
    
    inputs, targets, src_mask, labels, num_batches = get_batch(val_ds, batchsize, epoch_status, limited_seq_len)
    with torch.no_grad():
        for batch in range(num_batches):
            with torch.cuda.amp.autocast():
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
    validation_error.append(total_loss / (batchsize * num_batches))
    default_val_loss = total_loss / (batchsize - 1)
    val_loss = total_loss / (batchsize * num_batches)
    return default_val_loss, val_loss

best_val_loss = float('inf')
epochs = 40
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    epoch_status = round(epoch/epochs, 2)
    train(model)
    default_val_loss, val_loss = evaluate(model)
    elapsed = time.time() - epoch_start_time

    evolution_file.write(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'default validation loss {default_val_loss:5.2f}| validation loss {val_loss:5.2f} \n')
    evolution_file.flush()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        #torch.save(model, '/home/projects/cpr_10006/people/enrcop/models/bert_model_vanilla.pt')

    scheduler.step()

plt.plot(range(epochs), training_error, label='Training error')
plt.plot(range(epochs), validation_error, label='Validation error')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/projects/cpr_10006/people/enrcop/Figures/BERT_error.png')
evolution_file.close()


'''
test_loss = evaluate(best_model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)
'''
