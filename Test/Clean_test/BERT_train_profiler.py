#!/usr/bin/env python

import time
import torch
from torch import nn
import torch.nn.functional as F

from data_load_clean import *
from utils_clean import *
from architectures_clean import *

import cProfile
import pstats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with cProfile.Profile() as pr:
    # Define voabulary
    num_bins = 50000 
    vocab = get_vocab(num_bins)

    # Get Data specifying the directory where raw data is or we want to store
    files_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\dummy_data\\'
    #files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens/CLS_desc_rettime_tokens/'
    train_ds, val_ds = TrainingBERTDataLoader(files_dir, vocab, num_bins, training_percentage=0.5, validation_percentage=0.3, CLS_token=True, add_ret_time=True)
    #evolution_file = open('C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Dummy_results\\training_results.txt', "w")
    #evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/train_loss_cls_ret_halfsize.txt', "w")
    evolution_file = open('C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\dummy_results.txt', "w")

    # Define the model network for training BERT model

    model = TransformerModel(ntoken=len(vocab),  # size of vocabulary 
                            d_model= 10, #384, # embedding dimension
                            d_hid= 10, #384, # dimension of the feedforward network model in nn.TransformerEncoder
                            nhead= 2, #6, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                            nlayers=2, #6, # number of heads in nn.MultiheadAttention
                            activation=F.gelu, # activation function on the intermediate layer
                            dropout=0.1).to(device) # dropout probability

    # Define parameters for the training

    n_epochs = 2
    batchsize = 3
    num_batches = len(flatten_list(train_ds, 0)) // batchsize

    criterion = nn.CrossEntropyLoss()
    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-4) 
    # From https://github.com/ml-jku/DeepRC/blob/master/deeprc/training.py "eps needs to be at about 1e-4 to be numerically stable with 16 bit float"
    scheduler = BERT_scheduler(optimizer, learning_rate=lr, perc_warmup_steps=0.01, total_training_steps=n_epochs*num_batches)
    scheduler.step() # Initialize scheduler so lr starts warup from 0 as we want

    # Accelerating training time 
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None #lowering floating point precision when allowed
    limited_seq_len = 128
    perc_epochs_shorter = 0.9

    training_error = []; validation_error = []

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(1, n_epochs + 1):
        print('Epoch number: ', epoch)
        epoch_start_time = time.time()

        train_loss = BERT_train(model, optimizer, criterion, scheduler, dataset=train_ds, results_file=evolution_file, \
            batchsize = batchsize, current_epoch=epoch, total_epochs=n_epochs, limited_seq_len=limited_seq_len, \
            shorter_len_perc=perc_epochs_shorter, log_interval=1000, device=device, scaler=scaler)
    
        val_loss = BERT_evaluate(model, criterion, dataset=val_ds, results_file=evolution_file, batchsize=batchsize,\
            current_epoch=epoch, total_epochs=n_epochs, limited_seq_len=limited_seq_len, shorter_len_perc=perc_epochs_shorter, \
            start_time=epoch_start_time, device=device)

        training_error.append(train_loss)
        validation_error.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #torch.save({'epoch': epoch,
            #            'model_state_dict': model.state_dict(),
            #            'optimizer_state_dict': optimizer.state_dict(),
            #            'loss': train_loss
                        #}, 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Dummy_results\\bert_model.pt')
            #            }, '/home/projects/cpr_10006/people/enrcop/models/train/BERT_half_noCLS_norettime.pt')

    #save_plot = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Dummy_results\\test.png'
    #save_plot = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_train/BERT_half_noCLS_norettime.png'
    #save_plot = '/home/projects/cpr_10006/people/enrcop/Figures/dummy_plots/train.png'
    #plot_BERT_training_error(n_epochs, training_error, validation_error, pathway=save_plot)

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()

evolution_file.close()