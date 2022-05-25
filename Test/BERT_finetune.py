#!/usr/bin/env python

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy

from statistics import mean
import time
import matplotlib.pyplot as plt

from data_load import *
from utils import *
from architectures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define vocabulary (special tokens already considered) and type if classification layer
vocab = get_vocab(num_bins=50000)

class_layer = 'CNN'

# Define file where loss results will be saved and directory where sample files are found

#files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens_0.0125/no_CLS_no_desc_no_rettime_10000_tokens/'
#evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/BERT_finetune/BERT_small/finetune_loss_CNN_group2_no_cls_no_desc_no_rettime_0.0125_10000.txt', "w")
## LOCAL PATHWAY
files_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\dummy_data\\'
evolution_file = open('C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\dummy_results\\loss.txt', "w")

train_finetune_ds, val_finetune_ds, num_labels, min_scan_count = FineTuneBERTDataLoader(files_dir, vocab, training_percentage=0.02, validation_percentage=0.01, \
classification_layer=class_layer, class_param = 'Group2',
labels_path = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\test_data\\ALD_histological_scores.csv') # Group2 is equivalent to Healthy vs ALD

# Create the model network
# When saved properly with the General Checkpoint and decoder layer removal for fine tuning not done yet
# Define the model network for BERT model and load trained weights
'''
BERT_model = TransformerModel(ntoken = len(vocab), 
            d_model = 192, 
            d_hid = 192,
            nhead = 3,  
            nlayers = 3, 
            activation = F.gelu, 
            dropout = 0.1).to(device)

optimizer = optimizer = torch.optim.AdamW(BERT_model.parameters(), lr=1e-4, eps=1e-4)

checkpoint = torch.load('/home/projects/cpr_10006/people/enrcop/models/BERT_train/BERT_small/BERT_small_no_CLS_no_desc_no_rettime_0.0125_10000.pt', map_location=device)
#checkpoint = torch.load('C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\models\\BERT_small_no_CLS_no_desc_no_rettime_0.0125_10000.pt', map_location=device)
BERT_model.load_state_dict(checkpoint['model_state_dict'])

# Remove last layer used for training with masking
state_dict = copy.deepcopy(BERT_model.state_dict())
del state_dict['decoder.weight']
del state_dict['decoder.bias']
'''
# Define the model network for BERT model and load trained weights
new_BERT_model = BERT_trained(ntoken = len(vocab), 
            d_model = 192, 
            d_hid = 192,
            nhead = 3,  
            nlayers = 3, 
            activation = F.gelu, 
            dropout = 0.1).to(device)

#new_BERT_model.load_state_dict(state_dict)

# Load BERT weights
#model_weights = '/home/projects/cpr_10006/people/enrcop/models/BERT_train/BERT_vanilla_small/bert_vanilla_small_weights.pt'
model_weights = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\models\\bert_vanilla_small_weights.pt'
new_BERT_model.load_state_dict(torch.load(model_weights, map_location=device))

# This way BERT weights are frozen through the finetuning training
## Define new layers for fine-tuning and new model
attention_network = AttentionNetwork(n_input_features = 192, n_layers = 2, n_units = 32)

batchsize = 10
if class_layer == 'CNN':
    scans_count = min_scan_count // batchsize * batchsize # this is the number of scans that will be obtained after the BERT model to pass to CNN
    classification_layer = CNNClassificationLayer(num_labels=num_labels, heigth = scans_count, width = 192, kernel = 5, padding = 1)
    #print('CNN as classsification layer')
else:
    classification_layer = LinearClassificationLayer(d_model = 192, num_labels = num_labels)

model = FineTune_classification(attention_network, classification_layer)

criterion = nn.CrossEntropyLoss()
lr = 3e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#scaler = torch.cuda.amp.GradScaler()
#top_attention_perc = 0.1

training_error = []; validation_error = []; accuracy_evolution = []

best_val_loss = float('inf')
epochs = 20
best_model = None

for epoch in range(1, epochs + 1):
    print('Epoch: ', epoch, '/', epochs)
    epoch_start_time = time.time()

    #train_error, att_weights_matrix = BERT_finetune_train(new_BERT_model, model, optimizer, criterion, learning_rate=lr, 
    #dataset=train_finetune_ds, results_file=evolution_file, batchsize=batchsize, epoch=epoch, sample_interval = 25, device=device,
    #same_sample=True, scaler=None)

    #training_error.append(train_error)

    val_loss, acc = BERT_finetune_evaluate(new_BERT_model, model, classification_layer='CNN', criterion=criterion, dataset=val_finetune_ds, results_file=evolution_file, 
    batchsize=batchsize, current_epoch=epoch, start_time=epoch_start_time, device=device)
 
    validation_error.append(val_loss)
    accuracy_evolution.append(acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_accuracy = acc
        best_att_weights_matrix = update_best_att_matrix(att_weights_matrix)
        #torch.save(model.state_dict(), '/home/projects/cpr_10006/people/enrcop/models/BERT_finetune/BERT_small/bert_Group2_HPvsALD.pt')


error_plot_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_small/no_CLS_no_desc_no_rettime_0.0125_10000/Group2_HPvsALD/CNN_finetune_error.png'
att_weights_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_small/no_CLS_no_desc_no_rettime_0.0125_10000/Group2_HPvsALD/CNN_att_weights.png'

#dummy_dir = '/home/projects/cpr_10006/people/enrcop/PROD/dummy_results/'
#error_plot_pathway = dummy_dir + 'finetune_err.png'
#att_weights_pathway = dummy_dir + 'att_weights.png'

plot_finetuning_error(epochs, training_error, validation_error, pathway=error_plot_pathway)
plot_att_weights(best_att_weights_matrix, pathway=att_weights_pathway)

evolution_file.write('Accuracy evolution per epoch: [epoch, accuracy] \n')
for i, accuracy_epoch in enumerate(accuracy_evolution):
    evolution_file.write(f'  [Epoch {i:3d}:  {accuracy_epoch:3.2f}]  ')
evolution_file.write('\n')
evolution_file.write(f'Accuracy for model with lower validation loss {best_accuracy:3.2f}')
evolution_file.close()


