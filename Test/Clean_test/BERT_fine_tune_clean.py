#!/usr/bin/env python

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from statistics import mean
import time
import matplotlib.pyplot as plt

from data_load_clean import *
from utils_clean import *
from architectures_clean import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define vocabulary (special tokens already considered)
vocab = get_vocab(num_bins=50000)

# Define file where loss results will be saved and directory where sample files are found
#evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/finetuning_loss_vanilla.txt')
#files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens/scan_desc_tokens_CLS/'
# labels_path define correctly default DataLoader function in data_load_clean
##LOCAL PATHWAY
files_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\test_data\\' 
train_finetune_ds, val_finetune_ds, num_labels = FineTuneBERTDataLoader(files_dir, vocab, training_percentage=0.6, validation_percentage=0.3, labels_path=files_dir + 'ALD_histological_scores.csv')
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
#BERT_model.load_state_dict(torch.load('/home/projects/cpr_10006/people/enrcop/models/bert_vanilla_small_weights.pt'))
##LOCAL PATHWAY
model_weights = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\models\\bert_vanilla_small_weights.pt'
BERT_model.load_state_dict(torch.load(model_weights, map_location=device))
 # This way BERT weights are frozen through the finetuning training

## Define new layers for fine-tuning and new model
attention_network = AttentionNetwork(n_input_features = 192, n_layers = 2, n_units = 32)
classification_layer = ClassificationLayer(d_model = 192, num_labels = num_labels)

model = FineTuneBERT(attention_network, classification_layer).to(device)

batchsize = 16
criterion = nn.CrossEntropyLoss()
lr = 3e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scaler = torch.cuda.amp.GradScaler()

#top_attention_perc = 0.1

training_error = []; validation_error = []

def train(model: nn.Module) -> None:

    model.train()
    total_loss = 0
    log_interval = 1
    start_time = time.time()
    att_weights_matrix = []
    for sample in range(len(train_finetune_ds)):

        inputs, class_labels, num_batches = get_finetune_batch(train_finetune_ds[sample], batchsize, same_sample=True)
        
        #print('Input BERT size', inputs[0].size())
        #print('Class label size', class_labels[0].size())

        #num_class_spectra = top_attention_perc * len(inputs)

        inside_train_error = []
        hidden_vectors_sample = []

        for batch in range(num_batches):    
            hidden_vectors_batch = BERT_model(inputs[batch].to(device))
            hidden_vectors_sample.append(hidden_vectors_batch)
        #print('BERT output size: ', hidden_vectors_sample[0].size())
        #print('Attention layer input size: ', torch.cat(hidden_vectors_sample).size())
        att_weights, output = model(torch.cat(hidden_vectors_sample).to(device))
        #print('Final output size: ', output.size())
        
        att_weights_matrix.append(att_weights)
        '''
        max_size = max([len(sample) for sample in att_weights_matrix])
        best_att_weights_matrix = [tensor.tolist() for tensor in att_weights_matrix]
   
        for sample in best_att_weights_matrix:
            if len(sample) < max_size:
                sample.append([0]*(max_size - len(sample)))

        plt.matshow(best_att_weights_matrix)
        plt.xlabel('Scans / Spectra')
        plt.ylabel('Samples / Patients')
        plt.colorbar()
        plt.show()
        '''
        '''
        # If at some point we wanna take the x% most informative spectra
        _, used_spectra = torch.topk(att_weights, num_class_spectra)

        used_output = output[used_spectra].to(device)
        used_labels = class_labels[used_spectra].to(device)
        '''
        loss = criterion(torch.squeeze(output), torch.cat(class_labels))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

        if sample % log_interval == 0 and sample > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            inside_train_error.append(cur_loss)
            print(f'| epoch {epoch:3d} | {sample:2d}/{len(train_finetune_ds):1d} samples | '
            f'lr {lr:02.5f} | ms/sample {ms_per_batch:5.2f} | '
            f'loss {loss:5.2f} \n')
            #evolution_file.write(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
            #f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
            #f'loss {loss:5.2f} \n')     
            #evolution_file.flush()
            total_loss = 0
            start_time = time.time()
    
    return inside_train_error, att_weights_matrix

def evaluate(model: nn.Module) -> float:

    model.eval()
    total_loss = 0.
    # For validation, we don't just pass one sample at a time, but randomize
    inputs, class_labels, num_batches = get_finetune_batch(val_finetune_ds, batchsize, same_sample = False)
    
    with torch.no_grad():
        for batch in range(num_batches):
            hidden_vectors = BERT_model(inputs[batch].to(device))
            _, output = model(hidden_vectors.to(device))
            loss = criterion(output, class_labels[batch].to(device))
            total_loss += loss.item()
        
    val_loss = total_loss / num_batches
    validation_error.append(val_loss)
    return val_loss

best_val_loss = float('inf')
epochs = 5
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    inside_train_error, att_weights_matrix = train(model)
    training_error.append(mean(inside_train_error))
    val_loss = evaluate(model)
    elapsed = time.time() - epoch_start_time

    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'validation loss {val_loss:5.2f} \n')
    #evolution_file.write(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
    #      f'validation loss {val_loss:5.2f} \n')
    #evolution_file.flush()

    if val_loss < float('inf') : #best_val_loss:
        best_val_loss = val_loss
        # Save the attention weights for this best performing model
        max_size = max([len(sample) for sample in att_weights_matrix])
        best_att_weights_matrix = [tensor.tolist() for tensor in att_weights_matrix]
        for sample in best_att_weights_matrix:
            if len(sample) < max_size:
                sample.extend([[0]]*(max_size - len(sample)))

        #torch.save(model.state_dict(), '/home/projects/cpr_10006/people/enrcop/models/finetune_vanillabert_binary_HPvsALD.pt')

plt.plot(range(epochs), training_error, label='Training error')
plt.plot(range(epochs), validation_error, label='Validation error')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.getcwd() + '/finetune_error.png')

plt.matshow(best_att_weights_matrix)
plt.xlabel('Scans / Spectra')
plt.ylabel('Samples / Patients')
plt.colorbar()
plt.savefig(os.getcwd() + '/att_weights.png')
#plt.show()
#plt.savefig('/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune_vanilla_small.png')

#evolution_file.close()


