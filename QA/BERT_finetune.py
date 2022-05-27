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

# Define vocabulary (special tokens already considered) and type if classification layer
vocab = get_vocab(num_bins=10000)

# Define the architecture chosen for the fine-tuning
class_layer = 'CNN'
att_matrix = True # If CNN att_matrix must always be True

# Define pathway from where we get the tokens
files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens_0.0125/no_CLS_no_desc_no_rettime_10000_tokens/'

# Load BERT weights
model_checkpoint = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\models\\BERT_small_no_CLS_no_desc_no_rettime_0.0125_10000.pt'

# Define pathways when the different results are saved
evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/BERT_finetune/BERT_small/finetune_loss_CNN_group2_no_cls_no_desc_no_rettime_0.0125_10000.txt', "w")
error_plot_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_small/no_CLS_no_desc_no_rettime_0.0125_10000/Group2_HPvsALD/CNN_finetune_error.png'
att_weights_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_small/no_CLS_no_desc_no_rettime_0.0125_10000/Group2_HPvsALD/CNN_att_weights.png'
save_finetuning_model = '/home/projects/cpr_10006/people/enrcop/models/BERT_finetune/BERT_small/bert_Group2_HPvsALD.pt'

# Get the datasets for the fine-tuning
train_finetune_ds, val_finetune_ds, num_labels, min_scan_count = FineTuneBERTDataLoader(files_dir, 
                                                                vocab, 
                                                                training_percentage=0.5, 
                                                                validation_percentage=0.5, 
                                                                classification_layer=class_layer, 
                                                                class_param = 'Group2') # Group2 is equivalent to Healthy vs ALD

# Load the BERT model weights (Due to limited resources, this will be frozen)
new_BERT_model = load_BERT_model(model_checkpoint,
                                old_model = TransformerModel,
                                new_model=BERT_trained,
                                ntoken=len(vocab),
                                embed_size=192,
                                d_hid=192,
                                nhead=3,
                                nlayers=3,
                                device=device)

# Get classification layer architecture depending on our choices
class_model, BERT_fine_tune_train = define_architecture(class_layer=class_layer,
                                                att_matrix=att_matrix,
                                                n_input=192,
                                                num_labels=num_labels,
                                                n_layers_attention=2,
                                                n_units_attention=32,
                                                kernel_size=2,
                                                padding=1,
                                                n_layers_linear=None,
                                                n_units_linear=None)

# Define hyperparameters
batchsize = 2
criterion = nn.CrossEntropyLoss()
lr = 3e-5
optimizer = torch.optim.AdamW(class_model.parameters(), lr=lr)

#top_attention_perc = 0.1

training_error = []; validation_error = []; accuracy_evolution = []

best_val_loss = float('inf')
epochs = 3
best_model = None

for epoch in range(1, epochs + 1):
    print('Epoch: ', epoch, '/', epochs)
    epoch_start_time = time.time()
    
    train_error, att_weights_matrix = BERT_fine_tune_train(new_BERT_model, class_model, optimizer, criterion, learning_rate=lr, 
    dataset=train_finetune_ds, results_file=evolution_file, batchsize=batchsize, epoch=epoch, write_interval = 25, device=device,
    scaler=None)

    training_error.append(train_error)

    val_loss, acc = BERT_finetune_evaluate(new_BERT_model, class_model, att_matrix=att_matrix, class_layer=class_layer, criterion=criterion, 
    dataset=val_finetune_ds, results_file=evolution_file, batchsize=batchsize, current_epoch=epoch, start_time=epoch_start_time, device=device)
 
    validation_error.append(val_loss)
    accuracy_evolution.append(acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_accuracy = acc
        torch.save(class_model.state_dict(), save_finetuning_model)
        if att_matrix:
            best_att_weights_matrix = update_best_att_matrix(att_weights_matrix)
        

plot_finetuning_error(epochs, training_error, validation_error, pathway=error_plot_pathway)

if att_matrix:
    plot_att_weights(best_att_weights_matrix, pathway=att_weights_pathway)

evolution_file.write('Accuracy evolution per epoch: [epoch, accuracy] \n')
for i, accuracy_epoch in enumerate(accuracy_evolution):
    evolution_file.write(f'  [Epoch {i:3d}:  {accuracy_epoch:3.2f}]  ')
evolution_file.write('\n')
evolution_file.write(f'Accuracy for model with lower validation loss {best_accuracy:3.2f}')
evolution_file.close()


