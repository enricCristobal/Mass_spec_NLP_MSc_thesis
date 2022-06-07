#!/usr/bin/env python

import torch
from torch import nn

from data_load import *
from utils import *
from architectures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define vocabulary (special tokens already considered) and type if classification layer
vocab = get_vocab(num_bins=10000)

# Define pathways for the different uploads and savings
#files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens_0.0125/no_CLS_no_desc_no_rettime_10000_tokens/'
#evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/BERT_finetune/BERT_small/finetune_loss_CNN_group2_no_cls_no_desc_no_rettime_0.0125_10000.txt', "w")
## LOCAL PATHWAY
files_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\only_numbers_short\\'
evolution_file = open('C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\dummy_results\\loss.txt', "w")

save_baseline_model = '/home/projects/cpr_10006/people/enrcop/models/Baseline/VAE/basic.pt'
save_loss = '/home/projects/cpr_10006/people/enrcop/loss_files/VAE/vanilla_VAE_loss.txt'
save_latent_fig = '/home/projects/cpr_10006/people/enrcop/Figures/VAE/vanilla_VAE.png'
#save_model = '/home/projects/cpr_10006/people/enrcop/models/VAE/VAE_vanilla.pt'
save_model = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\dummy_results\\test_VAE_model.pt'

train_ds, val_ds, num_labels, min_scan_count = FineTuneBERTDataLoader(files_dir, 
                                                                vocab, 
                                                                training_percentage=0.8, 
                                                                validation_percentage=0.4, 
                                                                crop_input = True, 
                                                                class_param = 'Group2',
                                                                kleiner_type=None,
labels_path = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\test_data\\ALD_histological_scores.csv') # Group2 is equivalent to Healthy vs ALD

model = ConvVAE(n_channels=2,
                kernel=15,
                stride=5,
                padding=1,
                input_heigth=min_scan_count,
                input_width=512,
                latent_dim=2).to(device)

lr = 0.001
epochs = 3
batch_size = 2

optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.MSELoss()

training_loss = []
validation_loss = []
best_val_loss = float('inf')
best_model = None

for epoch in range(1, epochs+1):
    train_loss, latent_space_samples, labels = train_VAE(model, vocab, train_ds, device, optimizer, criterion, batchsize=batch_size)
    training_loss.append(train_loss)

    val_loss = val_VAE(model, vocab, val_ds, device, optimizer, criterion, batchsize=batch_size)
    validation_loss.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'epoch': epoch,
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'loss': train_loss
                    #}, 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Dummy_results\\bert_model.pt')
                    }, save_model)

    evolution_file.write(f'[Epoch {epoch:3d}]: Training loss: {train_loss:.4f} / Validation loss: {val_loss:.4f}  \n')

#plot_BERT_training_error(epochs, training_loss, validation_loss, save_loss)
#plt.clf()
#plot_VAE_embedding(latent_space_samples, labels, save_latent_fig)