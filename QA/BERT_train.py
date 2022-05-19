#!/usr/bin/env python

import time
import torch
from torch import nn
import torch.nn.functional as F

from data_load import *
from utils import *
from architectures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define voabulary
num_bins = 10000 
vocab = get_vocab(num_bins)

# Get Data specifying the directory where raw data is or we want to store
files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens_0.0125/no_CLS_no_desc_no_rettime_10000_tokens/'
##LOCALM PATHWAY!!:
#files_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\test_data\\'
#evolution_file = open('C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\loss_file.txt', "w")
evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/train_loss_no_cls_no_desc_no_ret_small_0.0125_10000.txt', "w")

#save_model = '/home/projects/cpr_10006/people/enrcop/models/BERT_train/BERT_small_no_CLS_no_desc_no_rettime_0.0125_10000.pt'
#save_plot = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_train/BERT_small_no_CLS_no_desc_no_rettime_0.0125_10000.png'
save_model = '/home/projects/cpr_10006/people/enrcop/PROD/dummy_results/model_train.pt'
save_plot = '/home/projects/cpr_10006/people/enrcop/PROD/dummy_results/plot_train.png'

train_ds, val_ds = TrainingBERTDataLoader(files_dir, 
                                        vocab, 
                                        training_percentage=0.05, 
                                        validation_percentage=0.01, 
                                        CLS_token=False, 
                                        add_ret_time=False,
                                        input_size=512,
                                        data_repetition=1)


# Define the model network for training BERT model

model = TransformerModel(ntoken=len(vocab),  # size of vocabulary 
                        d_model= 10, #192, # embedding dimension
                        d_hid= 10, #192, # dimension of the feedforward network model in nn.TransformerEncoder
                        nhead= 2, #3, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                        nlayers=2, #3,# number of heads in nn.MultiheadAttention
                        activation=F.gelu, # activation function on the intermediate layer
                        dropout=0.1).to(device) # dropout probability

# Define parameters for the training

n_epochs = 3
batchsize = 32
num_batches = len(flatten_list(train_ds, 0)) // batchsize

criterion = nn.CrossEntropyLoss()
lr = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-4) # From https://github.com/ml-jku/DeepRC/blob/master/deeprc/training.py "eps needs to be at about 1e-4 to be numerically stable with 16 bit float"
scheduler = BERT_scheduler(optimizer, learning_rate=lr, perc_warmup_steps=0.01, total_training_steps=n_epochs*num_batches)
scheduler.step() # Initialize scheduler so lr starts warup from 0 as we want

# Accelerating training time 
#scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None #lowering floating point precision when allowed
limited_seq_len = 128
perc_epochs_shorter = 0.9

training_error = []; validation_error = []

best_val_loss = float('inf')
best_model = None

early_stopping_counter = 0

for epoch in range(1, n_epochs + 1):
    print('Epoch: ', epoch , '/', n_epochs)
    epoch_start_time = time.time()

    train_loss = BERT_train(model, optimizer, criterion, scheduler, dataset=train_ds, results_file=evolution_file, \
        batchsize = batchsize, current_epoch=epoch, total_epochs=n_epochs, limited_seq_len=limited_seq_len, \
        shorter_len_perc=perc_epochs_shorter, log_interval=100, device=device, scaler=None)

    training_error.append(train_loss)

    val_loss = BERT_evaluate(model, criterion, dataset=val_ds, results_file=evolution_file, batchsize=batchsize,\
        current_epoch=epoch, total_epochs=n_epochs, limited_seq_len=limited_seq_len, shorter_len_perc=perc_epochs_shorter, \
        start_time=epoch_start_time, device=device)
    print(val_loss)
    validation_error.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save({'epoch': epoch,
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'loss': train_loss
                    #}, 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Dummy_results\\bert_model.pt')
                    }, save_model)
    else: 
        early_stopping_counter += 1
        if early_stopping_counter > 4:
            plot_BERT_training_error(epoch, training_error, validation_error, pathway=save_plot)
            exit()

plot_training_error(n_epochs, training_error, validation_error, pathway=save_plot)

evolution_file.close()
