#!/usr/bin/env python

import torch
from torch import nn
import time

from data_load import *
from utils import *
from architectures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define vocabulary (special tokens already considered) and type if classification layer
vocab = get_vocab(num_bins=50000)

# Define pathways for the different uploads and savings
#files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens_0.0125/no_CLS_no_desc_no_rettime_10000_tokens/'
#evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/BERT_finetune/BERT_small/finetune_loss_CNN_group2_no_cls_no_desc_no_rettime_0.0125_10000.txt', "w")
## LOCAL PATHWAY
files_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\no_CLS_desc_no_rettime_10000_tokens\\'
evolution_file = open('C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\dummy_results\\loss.txt', "w")

save_baseline_model = '/home/projects/cpr_10006/people/enrcop/models/Baseline/VAE/basic.pt'

train_finetune_ds, val_finetune_ds, num_labels, min_scan_count = FineTuneBERTDataLoader(files_dir, 
                                                                vocab, 
                                                                training_percentage=0.5, 
                                                                validation_percentage=0.5, 
                                                                classification_layer=None, 
                                                                class_param = 'nas_inflam',
labels_path = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\test_data\\ALD_histological_scores.csv') # Group2 is equivalent to Healthy vs ALD

