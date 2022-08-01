#!/usr/bin/env python

"""
Analaysis of output BERT embeddings for the detection
of more informative scans or some kind of structure.

Author - Enric Cristòbal Cóppulo
"""

import torch
import argparse

from data_load import *
from utils import *
from architectures import *

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(52)
    torch.cuda.manual_seed(52)

    # Define vocabulary (special tokens already considered) and type if classification layer
    vocab = get_vocab(num_bins=args.num_bins)

    class_param_long = 'kleiner_' + args.kleiner_type if args.kleiner_type else args.class_param 

    # Define pathways for the different uploads and savings
    files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/' + 'BERT_tokens_' + args.BERT_tokens + '/' + args.case +  '_' + str(args.num_bins) +  '_tokens/'

    # Load BERT weights
    model_weights = '/home/projects/cpr_10006/people/enrcop/models/BERT_train/' + args.BERT_type + '/' + class_param_long + '/' + args.case + '_' + args.BERT_tokens +  '_' + str(args.num_bins) + '_' + str(args.data_repetition) + '.pt'
    # Figures pathways
    save_plot_path = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_train/' + args.BERT_type + '/Heatmaps_detection/' + 'all_peaks_seen_data_'  + args.case + '_' + args.BERT_tokens +  '_' + str(args.num_bins) + '_' + str(args.data_repetition) + '_'


    _, val_finetune_ds, _, _ = FineTuneBERTDataLoader(files_dir, 
                                                    vocab, 
                                                    training_percentage=0, 
                                                    validation_percentage=0.5, 
                                                    crop_input=True, 
                                                    scan_detection=True,
                                                    class_param = args.class_param,
                                                    kleiner_type= args.kleiner_type)
    
    # Create the model network
    # Define the model network for BERT model and load trained weights

    if args.BERT_type == 'BERT_small':
        d_model = 192; d_hid = 192; nhead = 3; nlayers = 3
    elif args.BERT_type ==  'BERT_half':
        d_model = 384; d_hid = 384; nhead = 6; nlayers = 6

    # Define the model network for training BERT model
    BERT_model = BERT_trained(ntoken=len(vocab),  # size of vocabulary 
                            d_model=d_model, # embedding dimension
                            d_hid=d_hid, # dimension of the feedforward network model in nn.TransformerEncoder
                            nhead=nhead, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                            nlayers=nlayers,# number of heads in nn.MultiheadAttention
                            activation=F.gelu, # activation function on the intermediate layer
                            dropout=0.1).to(device) # dropout probability

    # Load BERT weights 
    checkpoint = torch.load(model_weights, map_location=device)
    BERT_model.load_state_dict(checkpoint['model_state_dict'], strict = False)

    # Define fine-tuning hyperparameters
    batchsize = args.batchsize

    BERT_scan_detection(BERT_model, dataset=val_finetune_ds, batchsize=batchsize, saveplot_path=save_plot_path, device=device)


if __name__ == '__main__':
  
    # create the parser
    parser = argparse.ArgumentParser(prog='BERT_scan_detection.py', formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=50, width=130), 
        description="Analysis of BERT embeddings ouput to detect potential informative scans.")
    
    parser.add_argument('--num_bins', help='Number of bins for the creation of the vocabulary', required=True, type=int)
    parser.add_argument('--BERT_tokens', help='Directory where to find the BERT tokens', required=True)
    parser.add_argument('--BERT_type', help='Type/Size of the BERT model to use (small, half)', required=True)
    parser.add_argument('--case', help='Combination of parameters for tokens used', required=True)
    parser.add_argument('--data_repetition', help='Number of times the different samples will be used masking scans in different random parts to increase datasize.', default=1, type=int)
    parser.add_argument('--batchsize', default=32, type=int)
    
    args = parser.parse_args()


    main(args)

