import torch.nn as nn
import torch
import math

from positional_encoding import PositionalEncoding
from attention import MultiHeadAttention
from feedforward import PositionWiseFeedForwardNetwork

class DecoderLayer(nn.Module) :
    def __init__(self, d_model, head, dropout, d_ff) :
        super(DecoderLayer, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.MaskedMultiHeadAttention = MultiHeadAttention(d_model, head)
        self.LayerNorm_1 = nn.LayerNorm(d_model)
        
        self.MultiHeadAttention = MultiHeadAttention(d_model, head)
        self.LayerNorm_2 = nn.LayerNorm(d_model)
        
        self.PositionWiseFeedForwardNetwork = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.LayerNorm_3 = nn.LayerNorm(d_model)
        self.Dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, encoder_output, look_ahead_mask, padding_mask) :
        '''
        x shape : [batch, seq_len, d_model]
        '''
        residual = x
        # Masked Multi-Head Attention
        Z, _ = self.MaskedMultiHeadAttention(x, x, x, look_ahead_mask) # output shape : [batch, seq_len, d_model]
        # Add & Norm
        x = self.Dropout(Z) + residual # output shape : [batch, seq_len, d_model]
        x = self.LayerNorm_1(x) # output shape : [batch, seq_len, d_model]
        
        residual = x
        # Multi-Head Attention
        Z, _ = self.MultiHeadAttention(q=x, k=encoder_output, v=encoder_output, mask=padding_mask) # output shape : [batch, seq_len, d_model]
        # Add & Norm
        x = self.Dropout(Z) + residual # output shape : [batch, seq_len, d_model]
        x = self.LayerNorm_2(x) # output shape : [batch, seq_len, d_model]
        
        residual = x
        
        # Feed Forward
        x = self.PositionWiseFeedForwardNetwork(x) # output shape : [batch, seq_len, d_model]
        # Add & Norm
        x = self.Dropout(x) + residual # output shape : [batch, seq_len, d_model]
        x = self.LayerNorm_3(x) # output shape : [batch, seq_len, d_model]
        
        return x
        
class Decoder(nn.Module) :
    def __init__(self, vocab_size, d_model, max_len, n, head, dropout, d_ff, device) :
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.PositionalEncoding = PositionalEncoding(max_len, d_model, device)
        
        self.Dropout = nn.Dropout(p=dropout)
        
        self.DecoderLayers = nn.ModuleList([DecoderLayer(d_model, head, dropout, d_ff) for _ in range(n)])
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, encoder_output, decoder_input, look_ahead_mask, padding_mask) :
        '''
        x shape : [batch, seq_len]
        '''
        input_embedding = self.input_embedding(decoder_input) * math.sqrt(self.d_model) # output shape : [batch, seq_len, d_model]
        positional_encoding = self.PositionalEncoding(decoder_input) # output shape : [batch, seq_len, d_model]
        x = self.Dropout(input_embedding + positional_encoding) # output shape : [batch, seq_len, d_model]
        
        # Decoder layer
        for decoder_layer in self.DecoderLayers :
            x = decoder_layer(x, encoder_output, look_ahead_mask, padding_mask) # output shape : [batch, seq_len, d_model]
        
        output = self.linear(x) # output : [batch, seq_len, vocab_size]
        
        return output