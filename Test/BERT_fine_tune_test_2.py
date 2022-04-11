#!/usr/bin/env python

from multiprocessing import BufferTooShort
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.vocab import build_vocab_from_iterator

import copy
import os

import pandas as pd

from statistics import mean
import math
import time
import random
import matplotlib.pyplot as plt

from data_load_test import data_load

## TODO!!!: Right now, these classes (TransformerModel and PositionalEncoding) need to be defined inside this code 
## to be able to update the model --> in the future all code should be unified in some way --> no need to have it here again

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout) #, activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_key_padding_mask: Tensor, shape [batch_size, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel_prefinetuning(nn.Module):
    # In this class we will get rid of the decoder layer, as we'll have deleted those layers from the state_dict from the training model
    # to be able to upload the weights

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, activation, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        #self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.decoder.bias.data.zero_()
        #self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_key_padding_mask: Tensor, shape [batch_size, seq_len] # In this case we get rid of this input because it is something
            we don't want anymore in our model, since now it is fine-tuning, not training 
            !!!! Important to consider when this class will have to be defined for both training and getting it for fine-tuning (maybe 
            it is not possible to have both)

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        #output = self.decoder(output)
        return output

# Define vocab required for the input of our fine-tuning

all_tokens = []
num_bins = 50000

for i in range(num_bins):
    all_tokens.append([str(i)])

vocab = build_vocab_from_iterator(all_tokens, specials=['[CLS]', '[SEP]', '[PAD]', '[EOS]', '[MASK]'])

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/finetuning_loss_vanilla.txt', "w")

## PROCESS TO GET RID OF LAST LAYER WHEN SAVING THE WHOLE MODEL

'''
training_model = torch.load('/home/projects/cpr_10006/people/enrcop/models/bert_vanilla_small.pt', map_location=device)

# In case of GPU:
# training_model = torch.load('/home/projects/cpr_10006/people/enrcop/models/bert_vanilla_small.pt)

print("Model's state_dict:")
for param_tensor in training_model.state_dict():
    print(param_tensor, "\t", training_model.state_dict()[param_tensor].size())

state_dict = copy.deepcopy(training_model.state_dict())
del state_dict['decoder.weight']
del state_dict['decoder.bias']

#torch.save(state_dict, '/home/projects/cpr_10006/people/enrcop/models/bert_vanilla_small_weights.pt')
'''

ntokens = len(vocab)  # size of vocabulary
emsize = 192 # embedding dimension
d_hid = 192 # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3 # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3 # number of heads in nn.MultiheadAttention
dropout = 0.1  # dropout probability
activation = F.gelu # activation function on the intermediate layer (For now with gelu is not working)

batchsize = 16

pre_model = TransformerModel_prefinetuning(ntokens, emsize, nhead, d_hid, nlayers, activation, dropout).to(device)
pre_model.load_state_dict(torch.load('/home/projects/cpr_10006/people/enrcop/models/bert_vanilla_small_weights.pt'))

'''
print("New model's state_dict:")
for param_tensor in pre_model.state_dict():
    print(param_tensor, "\t", pre_model.state_dict()[param_tensor].size())


# Print optimizer's state_dict

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
'''

#evolution_file = open('/home/projects/cpr_10006/people/enrcop/fine_tuning_evolution.txt', "w")

# Get the files for fine-tuning

## IMPORTANT!! WE NEED CLS TOKENS TO DO THE FINE-TUNING --> we nned to get those first


files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens/scan_desc_tokens_CLS/'

data = data_load(files_dir)

samples_list = os.listdir(files_dir)
finetune_percentage = 0.5 ## QUESTION: Do we need to use completely new files? Nope, in fact use the same as training
train_finetune_percentage = 0.8 # this is percentage of the files taking for finetunng used for training

file_limit_finetune = int(len(samples_list) * finetune_percentage)

#finetune_samples = samples_list[-file_limit_finetune:] # Get the last finetune_percentage of files within the directory

file_limit_training_filetune = int(len(finetune_samples) * train_finetune_percentage)
train_finetune_samples = finetune_samples[:file_limit_training_filetune]
val_finetune_samples = finetune_samples[file_limit_training_filetune:]


def fine_tune_ds(fine_tune_files, class_param = 'Group2', labels_path = '/home/projects/cpr_10006/projects/gala_ald/data/clinical_data/ALD_histological_scores.csv'):

    # Get dir of the '.raw' extension of file_tune_names to get a match with the csv dataframe later
    fine_tune_files = [file_name[:-4] for file_name in fine_tune_files]
    labels_df = pd.read_csv(labels_path, usecols=["File name", class_param])
    #print(labels_df['File name'][0])
    #if labels_df.index[labels_df['File name'] == '[473] 20190615_QE10_Evosep1_P0000005_LiNi_SA_Plate6_A2.htrms.PG.Quantity']:
    labels_df = labels_df[labels_df[class_param] != 'QC']
    
    # Due to some data format, we need to get rid of the first 4 characters of each File_name for later matching purposes with proper file name
    beginning_file_name = labels_df['File name'].str.find(']') + 2
    for i in range(len(labels_df)):
        labels_df['File name'].iloc[i] = labels_df['File name'].iloc[i][beginning_file_name.iloc[i]:-18] 
    
    labels_df['class_id'] = labels_df[class_param].factorize()[0]
    
    # If we want to save the original class name and know its id
    #category_id_df = df[[class_param, 'class_id']].drop_duplicates()
    #category_to_id = dict(category_id_df.values)
    
    labels_df = labels_df.loc[labels_df['File name'].isin(fine_tune_files)]

    return labels_df

labels_df = fine_tune_ds(finetune_samples)
#print(labels_df)

# TODO!!: Issues regarding imbalanced dataset!!! 

#print(len(labels_df.index[labels_df['class_id'] == 0]))
#print(len(labels_df.index[labels_df['class_id'] == 1]))

num_labels = len(labels_df['class_id'].unique())
#print('Num labels', num_labels)

# Create function to divide input, target, att_mask for each dataloader
flatten_list = lambda tensor, dimension:[spectra for sample in tensor for spectra in sample[dimension]]

def get_finetune_batch(data_ds, batchsize):
    # Transpose by .t() method because we want the size [seq_len, batch_size]
    # This function returns a list with len num_batches = len(input_data) // batchsize
    # where each element of this list contains a tensor of size [seq_len, batch_size]
    # Thus, looping through the whole list with len num_batches, we are going through 
    # the whole dataset, but passing it to the model in tensors of batch_size.

    input_data = flatten_list(data_ds, 0)
    labels_data = flatten_list(data_ds, 1)

    #print('In input_data: ', len(input_data))
    #print('In labels_data: ', len(labels_data))

    num_batches = len(input_data) // batchsize
    #print('num_batches: ', num_batches)
    input_batch = []; labels_batch = []; 
    indices = list(range(len(input_data)))

    for shuffling_times in range(10):
        random.shuffle(indices)

    for batch in range(num_batches):
        batch_indices = []
        for j in range(batchsize):
            batch_indices.append(indices.pop())

        input_batch.append(torch.stack([input_data[index] for index in batch_indices]).t())
        #att_mask_batch.append(torch.stack([att_mask[index] for index in batch_indices])) # because size passed needs to be (N, S), being N batchsize and S seq_len
        labels_batch.append(torch.stack([labels_data[index] for index in batch_indices]))
    #print('End input_batch: ', len(input_batch))
    #print('Size of each tensor inside input_batch: ', input_batch[0].size())

    return input_batch, labels_batch, num_batches

flatten_sample = lambda sample,dimension:[spectra for spectra in sample[dimension]]

def get_finetune_sample_batch(sample_data_ds, batchsize):
    # We consider the input is already just one sample
    input_data = flatten_sample(sample_data_ds, 0)
    labels_data = flatten_sample(sample_data_ds, 1)

    num_batches = len(input_data) // batchsize

    # We do same process of randomising a bit, so we don't follow the retention times of the experiment in order
    input_batch = []; labels_batch = []; 
    indices = list(range(len(input_data)))

    for shuffling_times in range(10):
        random.shuffle(indices)

    for batch in range(num_batches):
        batch_indices = []
        for j in range(batchsize):
            batch_indices.append(indices.pop())

        input_batch.append(torch.stack([input_data[index] for index in batch_indices]).t())
        labels_batch.append(torch.stack([labels_data[index] for index in batch_indices]))


    return input_batch, labels_batch, num_batches

# Get datasets:
def get_labels(df, samples_list):
    # Function to get the label for each sample (file) that is part of a list of interest with file names within a df
    labels = []; out_samples = []
    for i,f in enumerate(samples_list):
        if f[:-4] not in df.values: # In case some of the samples selected are QC
            out_samples.append(f) # so there are no mathing errors when getting val_finetune_ds because a label has been removed
        else:
            index = df.index[df['File name'] == f[:-4]]
            labels.append(df.at[index[0], 'class_id'])
    for removed_sample in out_samples: # Although it seems dummy, removing straight messed up the for loop
        samples_list.remove(removed_sample)
    return labels, samples_list

train_labels, train_finetune_samples = get_labels(labels_df, train_finetune_samples)
val_labels, val_finetune_samples= get_labels(labels_df, val_finetune_samples)


train_finetune_ds = [data.create_finetuning_dataset(f, vocab, train_labels[i]) for i,f in enumerate(train_finetune_samples)]
val_finetune_ds = [data.create_finetuning_dataset(f, vocab, val_labels[i]) for i,f in enumerate(val_finetune_samples)]

#print(train_finetune_ds[4][1][0], train_labels[4])
#print(train_finetune_ds[35][1][0], train_labels[35])
#print(train_finetune_ds[74][1][0], train_labels[74])

class FineTuneBERT(nn.Module):

    def __init__(self, pre_train_model, num_labels: int):
        super().__init__()
        self.pre_train_model = pre_train_model
        self.num_labels = num_labels
        d_model = pre_train_model.state_dict()['encoder.weight'].size()[1] #Get d_model from pre_train_model state_dict (encoder layer will always have this size)
        self.classification_layer = nn.Linear(d_model, num_labels) 
        
        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.classification_layer.bias.data.zero_()
        self.classification_layer.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: Tensor) -> Tensor:
        #print('Forward in: ', src.size())
        src = self.pre_train_model(src)
        #print('Out pre-train model: ', src.size())
        output = self.classification_layer(src)
        #print('Out fine-tune: ', output.size())
        return output

model = FineTuneBERT(pre_model, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
lr = 3e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#scaler = torch.cuda.amp.GradScaler()

training_error = []; validation_error = []

def train(model: nn.Module) -> None:

    model.train()
    total_loss = 0
    log_interval = 1000
    start_time = time.time()

    inputs, class_labels, num_batches = get_finetune_batch(train_finetune_ds, batchsize)
    #print('Input_size', inputs[0].size())
    #print('Class label size', class_labels[0].size())


    inside_train_error = []

    for batch in range(num_batches):

        output = model(inputs[batch].to(device))
        # Output will have size [512 x batchsize x class_labels] --> because each token will give a certain classification, that is why, we need to use only the 
        # first [CLS] token classification, it is this the one we are fine-tuning in this step --> output[0], optimizing then for this token to predict the correct
        # class_labels

        loss = criterion(output[0], class_labels[batch]) # can be done for all the samples of the batch at a time, but shape of input (output of the model in our case)
        # must be (N,C), being N=batchsize, C = number of classes, where for each scan we have the probability for each class    
        # and then class labels must have shape [N]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            inside_train_error.append(cur_loss)
            #print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
            #f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
            #f'loss {loss:5.2f} \n')
            evolution_file.write(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
            f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
            f'loss {loss:5.2f} \n')     
            evolution_file.flush()
            total_loss = 0
            start_time = time.time()
    training_error.append(mean(inside_train_error))

def sample_train(model: nn.Module) -> None:

    model.train()
    total_loss = 0
    sample_interval = 10
    start_time = time.time()
    
    for sample in range(len(train_finetune_ds)):

        inputs, class_labels, num_batches = get_finetune_batch(train_finetune_ds[sample], batchsize)
        #print('Input_size', inputs[0].size())
        #print('Class label size', class_labels[0].size())

        inside_train_error = []

        for batch in range(num_batches):

            output = model(inputs[batch].to(device))
            # Output will have size [512 x batchsize x class_labels] --> because each token will give a certain classification, that is why, we need to use only the 
            # first [CLS] token classification, it is this the one we are fine-tuning in this step --> output[0], optimizing then for this token to predict the correct
            # class_labels

            # can also be by mean or max along all hidden state vectors

            loss = criterion(output[0], class_labels[batch]) # can be done for all the samples of the batch at a time, but shape of input (output of the model in our case)
            # must be (N,C), being N=batchsize, C = number of classes, where for each scan we have the probability for each class    
            # and then class labels must have shape [N]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

        if sample % sample_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / sample_interval
            cur_loss = total_loss / sample_interval
            inside_train_error.append(cur_loss)
            #print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
            #f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
            #f'loss {loss:5.2f} \n')
            evolution_file.write(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
            f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
            f'loss {loss:5.2f} \n')     
            evolution_file.flush()
            total_loss = 0
            start_time = time.time()
        training_error.append(mean(inside_train_error))

def evaluate(model: nn.Module) -> float:

    model.eval()
    total_loss = 0.

    inputs, class_labels, num_batches = get_finetune_batch(val_finetune_ds, batchsize)

    with torch.no_grad():
        for batch in range(num_batches):
            output = model(inputs[batch].to(device))
            loss = criterion(output[0], class_labels[batch])

            total_loss += loss.item()
        
    val_loss = total_loss / num_batches
    validation_error.append(val_loss)
    return val_loss

best_val_loss = float('inf')
epochs = 5
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    sample_train(model)
    val_loss = evaluate(model)
    elapsed = time.time() - epoch_start_time

    evolution_file.write(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'validation loss {val_loss:5.2f} \n')
    evolution_file.flush()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '/home/projects/cpr_10006/people/enrcop/models/finetune_vanillabert_binary_HPvsALD.pt')

plt.plot(range(epochs), training_error, label='Training error')
plt.plot(range(epochs), validation_error, label='Validation error')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune_vanilla_small.png')

evolution_file.close()
