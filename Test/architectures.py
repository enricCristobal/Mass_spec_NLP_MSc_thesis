#!/usr/bin/env python

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math
import numpy as np

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, activation, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation)
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

class BERT_scheduler:

    def __init__(self, optimizer, learning_rate, total_training_steps):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self._step = 0
        self.warmup_steps = 10000 #perc_warmup_steps * total_training_steps (if we wanted to warmup a specific percentage of the training)
        self.total_training_steps = total_training_steps
        self._rate = 0

    def rate(self, current_step):
        # Idea from transformers package github https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/optimization.py
        """
        Return learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
        a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
        Args:
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            current_steps (`int`):
                Current step in the scheduler
        Return:
            Rate to multiply the learning rate for the current step.
        """
        if current_step < self.warmup_steps:
            lr_lambda = float(current_step) / float(max(1, self.warmup_steps))
        else:
            lr_lambda = max(
                0.0, float(self.total_training_steps - current_step) / float(max(1, self.total_training_steps - self.warmup_steps))
                )

        return lr_lambda
    
    def step(self):
        rate = self.rate(self._step)
        self._step += 1
        for p in self.optimizer.param_groups:
            p['lr'] = self.learning_rate * rate
        self._rate = rate
        self.optimizer.step()

    def get_lr(self):
        return self.learning_rate * self._rate


class BERT_trained(nn.Module):
    # In this class we will get rid of the decoder layer, as we'll have deleted those layers from the state_dict from the training model
    # to be able to upload the weights, and include the new attention and classification layers on top of it.

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, activation, dropout: float = 0.1):
        super(BERT_trained, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
       
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [batch_size, embedding_dim]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = torch.mean(output, dim = 0) # unify all embeddings by getting mean, instead of just using CLS token info 
        return output


class AttentionNetwork(nn.Module):
    # following code for AttentionNetwork obtained from https://github.com/ml-jku/DeepRC/blob/master/deeprc/architectures.py
    def __init__(self, n_input_features: int, n_layers: int = 2, n_units: int = 32):
        super(AttentionNetwork, self).__init__()
        
        self.n_attention_layers = n_layers
        self.n_units = n_units
       
        fc_attention = []
        for _ in range(self.n_attention_layers):
            att_linear = nn.Linear(n_input_features, self.n_units)
            att_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(att_linear.weight.shape)))
            fc_attention.append(att_linear)
            fc_attention.append(nn.SELU()) # from reference nn.SELU() used(tried with GeLU too)
            n_input_features = self.n_units
        
        att_linear = nn.Linear(n_input_features, 1)
        att_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(att_linear.weight.shape)))
        fc_attention.append(att_linear)
        self.attention_nn = torch.nn.Sequential(*fc_attention)
    
    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [# scans per sample, embedding_dim] (After applying torch.cat to all the embeddings of all scans per sample)
                Comment: Remember we will aplly mean over all the embeddings of all tokens per scan in the sample.
        Returns:
            attention_weights: Tensor of shape [#scans per sample, 1]
        """
        attention_weights = self.attention_nn(src)
        return attention_weights


class LinearClassificationLayer(nn.Module):
    def __init__(self, n_input_features: int, num_labels: int, n_layers: int = 2, n_units: int = 32):
        super(LinearClassificationLayer, self).__init__()
        
        self.input_features = n_input_features
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.n_units = n_units
        initrange = 0.1

        fc_linear = []
        for _ in range(self.n_layers):
            linear_layer = nn.Linear(n_input_features, n_units)
            linear_layer.bias.data.zero_()
            linear_layer.weight.data.uniform_(-initrange, initrange)
            fc_linear.append(linear_layer)
            fc_linear.append(nn.ReLU())
            n_input_features = n_units
        
        final_layer = nn.Linear(n_input_features, num_labels)
        final_layer.bias.data.zero_()
        final_layer.weight.data.uniform_(-initrange, initrange)
        fc_linear.append(final_layer)
        self.classification_layer = torch.nn.Sequential(*fc_linear)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [# scans per sample, embedding_dim]

        Returns:
            output: Tensor of shape [#scans per sample, num_labels]
        """

        output = self.classification_layer(src)
        return output


class CNNClassificationLayer(nn.Module):
    def __init__(self, num_labels: int, num_channels: int, kernel: int, padding: int, n_units: int): # might get idea from AttentionNetwork to make it dynamic
        super(CNNClassificationLayer, self).__init__()
        self.conv1 = nn.Conv2d(1,num_channels,kernel_size=kernel, padding=padding)
        self.pool = nn.MaxPool2d(2,2)
        self.BatchNorm2d = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, 1, kernel_size=kernel, padding=padding)
        self.fc1 = nn.LazyLinear(n_units)
        self.BatchNorm1d = nn.BatchNorm1d(n_units)
        self.fc2 = nn.Linear(n_units,num_labels)

    def forward(self, src: Tensor) -> Tensor:
        src = torch.unsqueeze(torch.unsqueeze(src, dim=0), dim= 0)
        src = self.pool(F.relu(self.conv1(src)))
        src = self.pool(F.relu(self.conv2(src)))
        src = src.view(1, -1) # flatten list so we have one prediction per sample in the end
        src = F.relu(self.fc1(src))
        output = self.fc2(src)
        return output


# In reality with this class we won't be fine-tuning BERT, but just the attention and classification layers, since it will
# be in those that we'll apply backpropagation while just using BERT as a frozen model
# Later, defining a class where also BERT model is included could be tried out --> Resource limits
class FineTune_classification(nn.Module):
    def __init__(self, attention_network, classification_layer):
        super(FineTune_classification, self).__init__()
        self.attention_nn = attention_network
        self.classification_nn = classification_layer
    
    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [# scans per sample, embedding_dim]

        Returns:
            output: Tensor of shape [#scans per sample, num_labels]
        """
        att_weights = self.attention_nn(src)
        att_weights_softmax = F.softmax(att_weights, dim=0)
        src_after_attention = src * att_weights_softmax
        output = self.classification_nn(src_after_attention)
        return att_weights_softmax, output


## TODO!!: Fine-tuning model that also updates BERT model weights (Not possible in Computerome for resource limits)
class FineTuneBERT(nn.Module):
    def __init__(self, BERT_model, attention_network, classification_layer):
        super(FineTuneBERT, self).__init__()
        self.BERT = BERT_model
        self.attention_nn = attention_network
        self.classification_nn = classification_layer

    def forward(self, src: Tensor) -> Tensor:
        hidden_vectors = self.BERT(src)
        ## TODO!! : aggregate hidden_vectors for attention layers (add new param to choose aggregation function?)
        att_weights = self.attention_nn(hidden_vectors)
        att_weights_softmax = F.softmax(att_weights, dim=0)
        src_after_attention = src * att_weights_softmax
        output = self.classification_nn(src_after_attention)
        return att_weights_softmax, output


# VAE as baseline

VAE_encoder_size = lambda input_size, kernel_size, stride, padding: int((input_size + 2* padding - kernel_size) / stride + 1)
output_padding = lambda output_size, input_size, kernel_size, stride, padding: output_size  - (input_size-1)*stride + 2*padding - kernel_size

class ConvVAE(nn.Module):
    def __init__(self, n_channels: int, kernel: int, stride: int, padding: int, input_heigth: int, input_width: int, latent_dim: int=2):
        super(ConvVAE, self).__init__()
    
        #Encoder 
        self.enc1 = nn.Conv2d(1, n_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.enc2 = nn.Conv2d(n_channels, n_channels*2, kernel_size=kernel, stride=stride, padding=padding)
        self.enc3 = nn.Conv2d(n_channels*2, 16, kernel_size=kernel, stride=stride, padding=padding)

        layer_dims = []
        # Output encoder size:
        heigth = input_heigth
        width = input_width
        for _ in range(3):
            heigth = VAE_encoder_size(heigth, kernel_size=kernel, stride=stride, padding=padding)
            width = VAE_encoder_size(width, kernel_size=kernel, stride=stride, padding=padding)
            layer_dims.append([heigth, width])
         
        #fully connected layers for learning representations
        self.fc1 = nn.LazyLinear(128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 16*layer_dims[2][0]*layer_dims[2][1])

        #Decoder
        output_padding_dec1 = output_padding(layer_dims[1][0], heigth, kernel, stride, padding)
        self.dec1 = nn.ConvTranspose2d(16, n_channels*2, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding_dec1)
        output_padding_dec2 = output_padding(layer_dims[0][0], layer_dims[1][0], kernel, stride, padding)
        self.dec2 = nn.ConvTranspose2d(n_channels*2, n_channels, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding_dec2)
        output_padding_dec3 = output_padding(input_heigth, layer_dims[0][0], kernel, stride, padding)
        self.dec3 = nn.ConvTranspose2d(n_channels, 1, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding_dec3)
    
    def reparametrize(self, mu, log_var):
        """
        mu: mean from the econder's latent space
        log_var: log variance from the encoder's latent space
        """
        # Reparametrixation trick: instead of x ~ N(mu, std),
        # x = mu + std * N(0,1)
        # and now we can compute gradients w.r.t mu and std
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + eps*std
        return sample 

    def forward(self, src):
        #Encoding
        src = F.relu(self.enc1(src))
        src = F.relu(self.enc2(src))
        src = F.relu(self.enc3(src))
        
        batch, _, heigth, width = src.shape   
        src = src.flatten().reshape(batch,-1)
        hidden = self.fc1(src)
        # get mu and log_var
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        #get latent vector thorugh reparametrization
        latent_space_samples = self.reparametrize(mu, log_var) #this is our latent representation
        z = self.fc2(latent_space_samples) 
        z = z.view(batch, 16, heigth, width)
        #Decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        reconstruction = torch.sigmoid(self.dec3(x))

        final_width = reconstruction.shape[3]
        if final_width < 512:
            reconstruction = F.pad(reconstruction, pad=(0,512-final_width))
        elif final_width > 512:
            reconstruction = reconstruction[:,:,:,:512]
        
        return latent_space_samples, reconstruction, mu, log_var