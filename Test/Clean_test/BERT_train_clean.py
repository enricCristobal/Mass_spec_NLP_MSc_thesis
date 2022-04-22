#!/usr/bin/env python

import time
from statistics import mean

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
    return cur_loss

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

'''
best_val_loss = float('inf')
epochs = 40
best_model = None
'''

evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/train_loss_cls_ret_halfsize.txt', "w")

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    epoch_status = round(epoch/epochs, 2)
    loss = train(model)
    default_val_loss, val_loss = evaluate(model)
    elapsed = time.time() - epoch_start_time

    evolution_file.write(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'default validation loss {default_val_loss:5.2f}| validation loss {val_loss:5.2f} \n')
    evolution_file.flush()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, '/home/projects/cpr_10006/people/enrcop/models/bert_model_vanilla.pt')

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
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f}')
print('=' * 89)
'''
