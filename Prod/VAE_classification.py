#!/usr/bin/env python

import torch
from torch import nn
import copy 

from data_load import *
from utils import *
from architectures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define vocabulary (special tokens already considered) and type if classification layer
vocab = get_vocab(num_bins=10000)

# Define pathways for the different uploads and savings
files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens_0.0125/no_CLS_no_desc_no_rettime_10000_tokens/'
evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/VAE/Classification/class_no_CLS_no_desc_no_rettime_10000.txt', "w")

# Upload the trained VAE encoder weigths
weights_pathway = '/home/projects/cpr_10006/people/enrcop/models/VAE/group2_no_CLS_no_desc_no_rettime_10000.pt'

# Pathways to save figures
save_loss = '/home/projects/cpr_10006/people/enrcop/Figures/VAE/Classification/no_CLS_no_desc_no_rettime_10000_loss.png'
ROC_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/VAE/Classification/no_CLS_no_desc_no_rettime_10000_ROC.png'
cm_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/VAE/Classification/no_CLS_no_desc_no_rettime_10000_cm.png'

# Define the datasets 
train_ds, val_ds, num_labels, min_scan_count = FineTuneBERTDataLoader(files_dir, 
                                                                vocab, 
                                                                training_percentage=0.7, 
                                                                validation_percentage=0.2, 
                                                                crop_input = True, 
                                                                class_param = 'Group2',
                                                                kleiner_type=None)

# Define the models and upload the weigths for the VAE
VAE_model = ConvVAE_encoder(n_channels=2,
                kernel=15,
                stride=5,
                padding=1,
                latent_dim=2).to(device)

checkpoint = torch.load(weights_pathway, map_location=device)
VAE_model.load_state_dict(checkpoint['model_state_dict'], strict = False)

classification_network = LinearClassificationLayer(n_input_features=2, num_labels=num_labels, n_layers=2, n_units=32)

#Define hyperparameters
epochs = 5
sample_size = 16
lr = 0.001
optimizer = torch.optim.Adam(classification_network.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')
best_model = None
training_loss = []; validation_loss = []; accuracy_evolution = []
for epoch in range(1, epochs+1):

    train_loss = train_class_VAE(VAE_model, classification_network, vocab, train_ds, device, optimizer, criterion, batchsize=sample_size)
    training_loss.append(train_loss)

    val_loss, accuracy, y_true, y_pred = val_class_VAE(VAE_model, classification_network, vocab, val_ds, device, optimizer, criterion, batchsize=sample_size)
    validation_loss.append(val_loss)
    accuracy_evolution.append(accuracy)

    evolution_file.write(f'[Epoch {epoch:3d}]: Training loss: {train_loss:.4f} / Validation loss: {val_loss:.4f}  \n')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_accuracy = accuracy
        best_y_pred = y_pred
        best_y_true = y_true

        
confusion_matrix, precision, recall, F1_score = get_metrics(best_y_true, best_y_pred, cm_pathway, True, ROC_fig_pathway=ROC_pathway)
plt.clf()
plot_finetuning_error(epochs, training_loss, validation_loss, pathway=save_loss)

evolution_file.write('Accuracy evolution per epoch: [epoch, accuracy] \n')
for i, accuracy_epoch in enumerate(accuracy_evolution):
    evolution_file.write(f'  [Epoch {i:3d}:  {accuracy_epoch:3.2f}]  ')
evolution_file.write('\n')
evolution_file.write(f'Accuracy for model with lower validation loss {best_accuracy:3.2f}')
evolution_file.close()


