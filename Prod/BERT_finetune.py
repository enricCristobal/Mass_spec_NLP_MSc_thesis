#!/usr/bin/env python

import torch
from torch import nn
import time

from data_load import *
from utils import *
from architectures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(52)
torch.cuda.manual_seed(52)

# Define vocabulary (special tokens already considered) and type if classification layer
vocab = get_vocab(num_bins=10000)

# Define the architecture for the classification layer for the fine-tuning: crop_input required if CNN
class_layer = 'Linear'
att_matrix = False # If CNN att_matrix must always be True
crop_input = True if att_matrix else False

# Define pathways for the different uploads and savings
files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens_0.0125/no_CLS_no_desc_no_rettime_10000_tokens/'
evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/BERT_finetune/BERT_small/finetune_loss_CNN_group2_no_cls_no_desc_no_rettime_0.0125_10000.txt', "w")

# Load BERT weights
model_weights = '/home/projects/cpr_10006/people/enrcop/models/BERT_train/BERT_small/BERT_small_no_CLS_no_desc_no_rettime_0.0125_10000_2.0.pt'

# Figures pathways
error_plot_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_small/no_CLS_no_desc_no_rettime_0.0125_10000/Group2_HPvsALD/CNN_finetune_error.png'
att_weights_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_small/no_CLS_no_desc_no_rettime_0.0125_10000/Group2_HPvsALD/CNN_att_weights.png'
ROC_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_small/no_CLS_no_desc_no_rettime_0.0125_10000/Group2_HPvsALD/ROC_curve.png'
cm_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_small/no_CLS_no_desc_no_rettime_0.0125_10000/Group2_HPvsALD/conf_matrix.png'

# Save fine-tune model
#save_finetuning_model = '/home/projects/cpr_10006/people/enrcop/models/BERT_finetune/BERT_small/bert_Group2_HPvsALD.pt'

train_finetune_ds, val_finetune_ds, num_labels, min_scan_count = FineTuneBERTDataLoader(files_dir, 
                                                                vocab, 
                                                                training_percentage=0.7, 
                                                                validation_percentage=0.2, 
                                                                crop_input=crop_input, 
                                                                class_param = 'Group2',
                                                                kleiner_type=None)

# Create the model network
# When saved properly with the General Checkpoint and decoder layer removal for fine tuning not done yet
# Define the model network for BERT model and load trained weights

BERT_model = BERT_trained(ntoken=len(vocab),  # size of vocabulary 
                    d_model= 192, # embedding dimension
                    d_hid= 192, # dimension of the feedforward network model in nn.TransformerEncoder
                    nhead= 3, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                    nlayers=3,# number of heads in nn.MultiheadAttention
                    activation=F.gelu, # activation function on the intermediate layer
                    dropout=0.1).to(device) # dropout probability

checkpoint = torch.load(model_weights, map_location=device)
BERT_model.load_state_dict(checkpoint['model_state_dict'], strict = False)

# This way BERT weights are frozen through the finetuning training
# Define classification network architecture

model, BERT_fine_tune_train = define_architecture(class_layer=class_layer,
                                                att_matrix=att_matrix,
                                                n_input=192,
                                                num_labels=num_labels,
                                                n_layers_attention=2,
                                                n_units_attention=24,
                                                num_channels=1,
                                                kernel_size=10,
                                                padding=1,
                                                n_units_linear_CNN=5,
                                                n_layers_linear=2,
                                                n_units_linear=4)

# Define fine-tuning hyperparameters
epochs = 2
batchsize = 16
lr = 3e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
#scaler = torch.cuda.amp.GradScaler()

best_val_loss = float('inf')
best_model = None
training_error = []; validation_error = []; accuracy_evolution = []

for epoch in range(1, epochs + 1):
    
    epoch_start_time = time.time()
    
    train_error, att_weights_matrix = BERT_fine_tune_train(BERT_model, model, optimizer, criterion, learning_rate=lr, 
    dataset=train_finetune_ds, results_file=evolution_file, batchsize=batchsize, epoch=epoch, write_interval = 25, device=device,
    scaler=None)

    training_error.append(train_error)

    val_loss, acc, y_true,y_pred = BERT_finetune_evaluate(BERT_model, model, att_matrix=att_matrix, class_layer=class_layer, criterion=criterion, 
    dataset=val_finetune_ds, results_file=evolution_file, batchsize=batchsize, current_epoch=epoch, start_time=epoch_start_time, device=device)
    
    validation_error.append(val_loss)
    accuracy_evolution.append(acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_accuracy = acc
        best_y_pred = y_pred
        best_y_true = y_true
        #torch.save(model.state_dict(), save_finetuning_model)
        if att_matrix:
            best_att_weights_matrix = update_best_att_matrix(att_weights_matrix)
        
confusion_matrix, precision, recall, F1_score = get_metrics(best_y_true, best_y_pred, cm_pathway, ROC_curve=True, ROC_fig_pathway=ROC_pathway)
plot_finetuning_error(epochs, training_error, validation_error, pathway=error_plot_pathway)

if att_matrix:
    plot_att_weights(best_att_weights_matrix, pathway=att_weights_pathway)

evolution_file.write('Accuracy evolution per epoch: [epoch, accuracy] \n')
for i, accuracy_epoch in enumerate(accuracy_evolution):
    evolution_file.write(f'  [Epoch {i:3d}:  {accuracy_epoch:3.2f}]  ')
evolution_file.write('\n')
evolution_file.write(f'Accuracy for model with lower validation loss {best_accuracy:3.2f}')
evolution_file.close()