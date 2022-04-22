#!/usr/bin/env python

import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from data_load_clean import *
from utils_clean import *
from architectures_clean import *

import math
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define voabulary
num_bins = 50000 
vocab = get_vocab(num_bins)

# Get Data specifying the directory where raw data is or we want to store
files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens/scan_desc_tokens_CLS/'

train_ds, val_ds = TrainingBERTDataLoader(files_dir, vocab, num_bins, training_percentage=0.7, validation_percentage=0.2, CLS_token=True)

# Define the model network for training BERT model

model = TransformerModel(ntoken=len(vocab),  # size of vocabulary 
                        d_model=384, # embedding dimension
                        d_hid=384, # dimension of the feedforward network model in nn.TransformerEncoder
                        nhead=6, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                        nlayers=6, # number of heads in nn.MultiheadAttention
                        activation=F.gelu, # activation function on the intermediate layer
                        dropout=0.1).to(device) # dropout probability

criterion = nn.CrossEntropyLoss()
lr = 1e-4  # learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-4) # From https://github.com/ml-jku/DeepRC/blob/master/deeprc/training.py "eps needs to be at about 1e-4 to be numerically stable with 16 bit float"


best_val_loss = float('inf')
epochs = 40
best_model = None

input_data = flatten_list(train_ds, 0)    
num_batches = len(input_data) // batchsize

scheduler = BERT_scheduler(optimizer, warmup_steps=10000, total_training_steps=epochs*num_batches)
scheduler.step() # Initialize scheduler so lr starts warup from 0 as we want

scaler = torch.cuda.amp.GradScaler()

training_error = []; validation_error = []

batchsize = 24
limited_seq_len = 128
perc_epochs_shorter_seq_len = 0.9

'''
best_val_loss = float('inf')
epochs = 40
best_model = None
'''

evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/train_loss_cls_ret_halfsize.txt', "w")

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    
    mean_error, loss = BERT_train(model, optimizer, criterion, scheduler, dataset=train_ds, results_file=evolution_file, batchsize = batchsize,
    current_epoch=epoch, total_epochs=epochs, limited_seq_len=limited_seq_len, log_interval=1000, device=device, scaler=scaler)
   
    default_val_loss, val_loss = BERT_evaluate(model, criterion, dataset=val_ds, results_file=evolution_file, batchsize=batchsize,
    current_epoch=epoch, total_epochs=epochs, limited_seq_len=limited_seq_len, start_time=epoch_start_time, device=device)

    training_error.append(mean_error)
    validation_error.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, '/home/projects/cpr_10006/people/enrcop/models/bert_model_vanilla.pt')

    scheduler.step()

plot_BERT_training_error(epochs, training_error, validation_error, pathway='/home/projects/cpr_10006/people/enrcop/Figures/BERT_error.png')

evolution_file.close()

'''
test_loss = evaluate(best_model, test_data)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f}')
print('=' * 89)
'''
