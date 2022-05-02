#!/usr/bin/env python

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from statistics import mean
import time
import matplotlib.pyplot as plt

from data_load import *
from utils import *
from architectures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define vocabulary (special tokens already considered)
vocab = get_vocab(num_bins=50000)

# Define file where loss results will be saved and directory where sample files are found
evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/finetuning_loss_vanilla.txt')
files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens/CLS_desc_rettime_tokens/'
# !!labels_path define correctly default DataLoader function in data_load_clean
##LOCAL PATHWAY
#files_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\test_data\\' 
train_finetune_ds, val_finetune_ds, num_labels = FineTuneBERTDataLoader(files_dir, vocab, training_percentage=0.1, validation_percentage=0.05, \
class_param = 'Group2') # Group2 is equivalent to Healthy vs ALD

# Create the model network
#n_input_features = pre_model.state_dict()['encoder.weight'].size()[1] #Get d_model from BERT state_dict (encoder layer will always have this size)
# Old hyperparameters to load the BERT model

# Define the model network for BERT model and load trained weights
BERT_model = BERT_trained(ntoken = len(vocab), 
            d_model = 192, 
            d_hid = 192,
            nhead = 3,  
            nlayers = 3, 
            activation = F.gelu, 
            dropout = 0.1).to(device)

# Load BERT weights
##LOCAL PATHWAY
#model_weights = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\models\\bert_vanilla_small_weights.pt'
model_weights = '/home/projects/cpr_10006/people/enrcop/models/BERT_train/BERT_vanilla_small/bert_vanilla_small_weights.pt'
BERT_model.load_state_dict(torch.load(model_weights, map_location=device))
 # This way BERT weights are frozen through the finetuning training

## Define new layers for fine-tuning and new model
attention_network = AttentionNetwork(n_input_features = 192, n_layers = 2, n_units = 32)
classification_layer = ClassificationLayer(d_model = 192, num_labels = num_labels)

model = FineTune_classification(attention_network, classification_layer).to(device)

batchsize = 2
criterion = nn.CrossEntropyLoss()
lr = 3e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler()

#top_attention_perc = 0.1

training_error = []; validation_error = []

best_val_loss = float('inf')
epochs = 5
best_model = None

for epoch in range(1, epochs + 1):
    #print('Epoch: ', epoch, '/', epochs +1)
    epoch_start_time = time.time()

    inside_train_error, att_weights_matrix = BERT_finetune_train(BERT_model, model, optimizer, criterion, learning_rate=lr, 
    dataset=train_finetune_ds, results_file=evolution_file, batchsize=batchsize, epoch=epoch, log_interval = 1000, device=device,
    same_sample=True, scaler=None)

    val_loss = BERT_finetune_evaluate(BERT_model, model, criterion, dataset=val_finetune_ds, batchsize=batchsize, device=device)

    training_error.append(mean(inside_train_error))
    validation_error.append(val_loss)

    if val_loss < best_val_loss:
        print('Model improved')
        best_val_loss = val_loss
        best_att_weights_matrix = update_best_att_matrix(att_weights_matrix)
       #torch.save(model.state_dict(), '/home/projects/cpr_10006/people/enrcop/models/BERT_finetune/BERT_vanilla/bert_Group2_HPvsALD.pt')

#error_plot_pathway = os.getcwd() + '/finetune_error.png'
#att_weights_pathway = os.getcwd() + '/att_weights.png'
#error_plot_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_vanilla/Group2_HPvsALD/finetune_error2.0.png'
#att_weights_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_vanilla/Group2_HPvsALD/att_weights_mean2.0.png'
error_plot_pathway = '/home/projects/cpr_10006/people/enrcop/TEST/clean_code/test_figures/finetune_error.png'
att_weights_pathway = '/home/projects/cpr_10006/people/enrcop/TEST/clean_code/test_figures/att_weights.png'

plot_finetuning_error(epochs, training_error, validation_error, pathway=error_plot_pathway)
plot_att_weights(best_att_weights_matrix, pathway=att_weights_pathway)

evolution_file.close()


