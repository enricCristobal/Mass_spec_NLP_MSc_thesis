#!/usr/bin/env python

import torch
from torch import nn
import copy 

from data_load import *
from utils import *
from architectures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define vocabulary (special tokens already considered) and type if classification layer
vocab = get_vocab(num_bins=50000)

# Define pathways for the different uploads and savings
#files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens_0.0125/no_CLS_no_desc_no_rettime_10000_tokens/'
#evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/VAE/VAE_vanilla', "w")
## LOCAL PATHWAY
files_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\only_numbers_short\\'
evolution_file = open('C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\dummy_results\\loss.txt', "w")

save_loss = '/home/projects/cpr_10006/people/enrcop/Figures/VAE/vanilla_VAE_loss.png'
ROC_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/VAE/ROC_VAE.png'

weights_pathway = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\dummy_results\\test_VAE_model.pt'

train_ds, val_ds, num_labels, min_scan_count = FineTuneBERTDataLoader(files_dir, 
                                                                vocab, 
                                                                training_percentage=0, 
                                                                validation_percentage=1, 
                                                                crop_input = True, 
                                                                class_param = 'Group2',
                                                                kleiner_type=None,
labels_path = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\test_data\\ALD_histological_scores.csv') # Group2 is equivalent to Healthy vs ALD

VAE_model = ConvVAE_encoder(n_channels=2,
                kernel=15,
                stride=5,
                padding=1,
                latent_dim=2).to(device)

checkpoint = torch.load(weights_pathway, map_location=device)
VAE_model.load_state_dict(checkpoint['model_state_dict'], strict = False)

classification_network = LinearClassificationLayer(n_input_features=2, num_labels=num_labels, n_layers=2, n_units=32)
lr = 3e-5
epochs = 3
sample_size = 2

optimizer = torch.optim.Adam(classification_network.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()

training_loss = []; validation_loss = []; accuracy_evolution = []

best_val_loss = float('inf')
best_model = None

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

        
confusion_matrix, precision, recall, F1_score = get_metrics(best_y_true, best_y_pred, True, ROC_fig_pathway=None)
plt.clf()
plot_finetuning_error(epochs, training_loss, validation_loss, pathway=None)

evolution_file.write('Accuracy evolution per epoch: [epoch, accuracy] \n')
for i, accuracy_epoch in enumerate(accuracy_evolution):
    evolution_file.write(f'  [Epoch {i:3d}:  {accuracy_epoch:3.2f}]  ')
evolution_file.write('\n')
evolution_file.write(f'Accuracy for model with lower validation loss {best_accuracy:3.2f}')
evolution_file.close()


