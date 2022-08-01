#!/usr/bin/env python

"""
Main file for the training of the BERT model

Author - Enric Cristòbal Cóppulo
"""

import time
import torch
from torch import nn
import torch.nn.functional as F
import argparse

from data_load import *
from utils import *
from architectures import *


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(52)
    torch.cuda.manual_seed(52)

    # Define vocabulary
    vocab = get_vocab(num_bins=args.num_bins)

    # Get Data specifying the directory where raw data is or we want to store
    files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/' + 'BERT_tokens_' + args.BERT_tokens + '/' + args.case + '_' + str(args.num_bins) + '_tokens/' #BERT_tokens_0.0125/no_CLS_no_desc_no_rettime_10000_tokens/'
    evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/BERT_train/' + args.BERT_type + '/' + args.case + '_' + args.BERT_tokens + '_' + str(args.num_bins) + '_' + str(args.data_repetition) + '.txt', "w") # train_loss_no_cls_no_desc_no_ret_small_0.0125_10000_2.0.txt', "w")

    # Save model and training losses
    save_model = '/home/projects/cpr_10006/people/enrcop/models/BERT_train/' + args.BERT_type + '/' + args.case + '_' + args.BERT_tokens + '_' + str(args.num_bins) + '_' + str(args.data_repetition) + '.pt' 
    save_plot = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_train/' + args.BERT_type + '/' + args.case + '_' + args.BERT_tokens + '_' + str(args.num_bins) + '_' + str(args.data_repetition) + '.png'

    # Define the datasets
    train_ds, val_ds = TrainingBERTDataLoader(files_dir, 
                                            vocab, 
                                            training_percentage=0.7, 
                                            validation_percentage=0.2, 
                                            CLS_token=args.CLS_token, #True
                                            add_ret_time=args.add_ret_time, #True
                                            input_size=args.input_size, #512
                                            data_repetition=args.data_repetition)

    # Define sizes of the models depending on the type defined
    if args.BERT_type == 'BERT_small':
        d_model = 192; d_hid = 192; nhead = 3; nlayers = 3
    elif args.BERT_type ==  'BERT_half':
        d_model = 384; d_hid = 384; nhead = 6; nlayers = 6

    # Define the model network for training BERT model
    BERT_model = TransformerModel(ntoken=len(vocab),  # size of vocabulary 
                            d_model=d_model, # embedding dimension
                            d_hid=d_hid, # dimension of the feedforward network model in nn.TransformerEncoder
                            nhead=nhead, # Number of heads in Multi-head attention
                            nlayers=nlayers,# Number of sub-encoder-layers in the encoder
                            activation=F.gelu, # activation function on the intermediate layer
                            dropout=0.1).to(device) # dropout probability

    # Define parameters for the training
    n_epochs = args.n_epochs
    batchsize = args.batchsize
    criterion = nn.CrossEntropyLoss()
    lr = 1e-4
    optimizer = torch.optim.AdamW(BERT_model.parameters(), lr=lr, eps=1e-4) 

    num_batches = len(flatten_list(train_ds, 0)) // batchsize
    scheduler = BERT_scheduler(optimizer, learning_rate=lr, total_training_steps=n_epochs*num_batches)
    scheduler.step() # Initialize scheduler so lr starts warup from 0 as we want

    # Accelerating training time 
    #scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None #lowering floating point precision when allowed
    limited_seq_len = args.limited_seq_len
    perc_epochs_shorter = args.perc_epochs_shorter
    best_model = None
    best_val_loss = float('inf')
    early_stopping_counter = 0
    training_error = []; validation_error = []
    
    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        train_loss = BERT_train(BERT_model, optimizer, criterion, scheduler, dataset=train_ds, results_file=evolution_file, 
            batchsize = batchsize, current_epoch=epoch, total_epochs=n_epochs, limited_seq_len=limited_seq_len, 
            shorter_len_perc=perc_epochs_shorter, log_interval=1000, device=device, scaler=None)
      
        training_error.append(train_loss)
        val_loss = BERT_evaluate(BERT_model, criterion, dataset=val_ds, results_file=evolution_file, batchsize=batchsize,
            current_epoch=epoch, total_epochs=n_epochs, limited_seq_len=limited_seq_len, shorter_len_perc=perc_epochs_shorter, 
            start_time=epoch_start_time, device=device)
        
        validation_error.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = BERT_model
            early_stopping_counter = 0

            # If we want to ensure to save the whole model everytime the val_loss is improved
            torch.save({'epoch': epoch,
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss
                        }, save_model)
            
        else: 
            early_stopping_counter += 1
            if early_stopping_counter > 3:
                torch.save({'epoch': epoch,
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss
                        }, save_model)

                plot_BERT_training_error(epoch, training_error, validation_error, pathway=save_plot)
                exit()

    # If we want to save model at the end of the simulation:
    #torch.save({'epoch': epoch,
    #                'model_state_dict': best_model.state_dict(),
    #                'optimizer_state_dict': optimizer.state_dict(),
    #                'loss': train_loss
    #                   }, save_model)

    # Plot training and validation error among epochs for BERT training
    plot_BERT_training_error(n_epochs, training_error, validation_error, pathway=save_plot)

    evolution_file.close()


if __name__ == '__main__':
    
    # create the parser
    parser = argparse.ArgumentParser(prog='BERT_train.py', formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=50, width=130), 
        description='''Train BERT in an unsupervised manner.''')
    
    parser.add_argument('--num_bins', help='Number of bins for the creation of the vocabulary', required=True, type=int)
    parser.add_argument('--BERT_tokens', help='Directory where to find the BERT tokens', required=True)
    parser.add_argument('--BERT_type', help='Type/Size of the BERT model to use (small, half)', required=True)
    parser.add_argument('--case', help='Combination of parameters for tokens used', required=True)
    parser.add_argument('--CLS_token', help='Boolean defining if the sentences contain [CLS] toekn at the beginning', type=bool, required=True)
    parser.add_argument('--add_ret_time', help='Boolean defining if the sentences ontain the retention time and a [SEP] token at the beginning', type=bool, required=True)
    parser.add_argument('--input_size', help='Sentence lentgh that will be introdued to the BERT models', default=512, type=int)
    parser.add_argument('--data_repetition', help='Number of times the different samples will be used masking scans in different random parts to increase datasize.', default=1, type=int)
    parser.add_argument('--n_epochs', help='number epochs', default=40, type=int)
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--limited_seq_len', help='Reeduction of sentence length at the beginning of the training to reduce training time', default=128, type=int)
    parser.add_argument('--perc_epochs_shorter', help='Percentage training epochs where the shorter sequence length will be used', default=0.9, type=float)

    args = parser.parse_args()

    main(args)
