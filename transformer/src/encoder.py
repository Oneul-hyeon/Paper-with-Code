import torch.nn as nn
import torch
import math

from positional_encoding import PositionalEncoding
from attention import MultiHeadAttention
from feedforward import PositionWiseFeedForwardNetwork

class EncoderLayer(nn.Module) :
    def __init__(self, d_model, head, dropout, d_ff) :
        super(EncoderLayer, self).__init__()
        
        self.d_model = d_model
        self.head = head
        
        self.MultiHeadAttention = MultiHeadAttention(d_model=d_model, head=head)
        
        self.LayerNorm_1 = nn.LayerNorm(d_model)
        self.LayerNorm_2 = nn.LayerNorm(d_model)
        
        self.Dropout = nn.Dropout(p=dropout)
        self.PositionWiseFeedForwardNetwork = PositionWiseFeedForwardNetwork(d_model, d_ff)
        
    def forward(self, x, padding_mask) :
        residual = x

        # Multi-Head Attention
        '''
        Z shape : [batch, seq_len, d_model]
        attention_score : [batch, head, seq_len, seq_len]
        '''
        Z, _ = self.MultiHeadAttention(x, x, x, padding_mask)
        # add & norm
        x = self.Dropout(Z) + residual # output shape : [batch, seq_len, d_model]
        x = self.LayerNorm_1(x) # output shape : [batch, seq_len, d_model]
        
        residual = x
        
        # feedforward
        x = self.PositionWiseFeedForwardNetwork(x) # output shape : [batch, seq_len, d_model]
        
        # add & norm
        x = self.Dropout(x) + residual # output shape : [batch, seq_len, d_model]
        x = self.LayerNorm_2(x) # output shape : [batch, seq_len, d_model]
        
        return x

class Encoder(nn.Module) :
    def __init__(self, vocab_size, d_model, max_len, n, head, dropout, d_ff, device) :
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_len, d_model, device)
        self.Dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=d_model, head=head, dropout=dropout, d_ff=d_ff) for _ in range(n)])
        self.device = device
        
    def forward(self, x, padding_mask) :
        input_embedding = self.input_embedding(x) * math.sqrt(self.d_model) # output shape : (batch, seq_len, d_model)
        positional_encoding = self.positional_encoding(x) # output shape : (batch, seq_len, d_model)
        
        # encoder layer 입력
        x = self.Dropout(input_embedding + positional_encoding) # output shape : (batch, seq_len, d_model)
        
        # encoder layers 처리
        '''
        x shape : [batch, seq_len, d_model]
        self_attention shape : [batch, head, seq_len, seq_len]
        '''
        for encoder_layer in self.encoder_layers :
            x = encoder_layer(x, padding_mask)
        
        return x