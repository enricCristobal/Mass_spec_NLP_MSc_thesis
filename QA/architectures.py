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


class BERT_trained(nn.Module):
    # In this class we will get rid of the decoder layer, as we'll have deleted those layers from the state_dict from the training model
    # to be able to upload the weights

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, activation, dropout: float = 0.1):
        super(BERT_trained, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_key_padding_mask: Tensor, shape [batch_size, seq_len] # In this case we get rid of this input because it is something
            we don't want anymore in our model, since now it is fine-tuning, not training 

        Returns:
            output Tensor of shape [d_model, 1], since we apply the mean over all scans
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
            fc_attention.append(nn.SELU()) # from reference (might have to change to GeLU)
            n_input_features = self.n_units
        
        att_linear = nn.Linear(n_input_features, 1)
        att_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(att_linear.weight.shape)))
        fc_attention.append(att_linear)
        self.attention_nn = torch.nn.Sequential(*fc_attention)
    
    def forward(self, src: Tensor) -> Tensor:
        attention_weights = self.attention_nn(src)
        return attention_weights


class ClassificationLayer(nn.Module):
    def __init__(self, d_model: int, num_labels: int):
        super(ClassificationLayer, self).__init__()
        self.num_labels = num_labels
        self.classification_layer = nn.Linear(d_model, num_labels) 
        
        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.classification_layer.bias.data.zero_()
        self.classification_layer.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: Tensor) -> Tensor:
        output = self.classification_layer(src)
        return output


# In reality with this class we won't be fine-tuning BERT, but just the attention and classification layers, since it will
# be in those that we'll apply backpropagation while just using BERT as a frozen model
# Later, defining a class where also BERT model is included could be tried out
class FineTuneBERT(nn.Module):
    def __init__(self, attention_network, classification_layer):
        super(FineTuneBERT, self).__init__()
        self.attention_nn = attention_network
        self.classification_nn = classification_layer
    
    def forward(self, src: Tensor) -> Tensor:
        att_weights = self.attention_nn(src)
        att_weights_softmax = F.softmax(att_weights, dim=0)
        src_after_attention = src * att_weights_softmax
        output = self.classification_nn(src_after_attention)
        return att_weights, output




