import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module) :
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, src_max_len, tgt_max_len, n, head, dropout, d_ff, padding_idx, device) :
        super(Transformer, self).__init__()
        
        self.padding_idx = padding_idx
        self.device = device
        
        self.Encoder = Encoder(vocab_size=src_vocab_size,
                               d_model=d_model,
                               max_len=src_max_len,
                               n=n,
                               head=head,
                               dropout=dropout,
                               d_ff=d_ff,
                               device=device)
        self.Decoder = Decoder(vocab_size=tgt_vocab_size,
                               d_model=d_model,
                               max_len=tgt_max_len,
                               n=n,
                               head=head,
                               dropout=dropout,
                               d_ff=d_ff,
                               device=device)
        
    def forward(self, encoder_input, decoder_input) :
        # mask 생성
        encoder_mask = self.padding_mask(encoder_input, encoder_input)
        encoder_decoder_mask = self.padding_mask(decoder_input, encoder_input)
        look_ahead_mask = self.padding_mask(decoder_input, decoder_input) * self.look_ahead_mask(decoder_input)
        encoder_out = self.Encoder(encoder_input, encoder_mask)
        decoder_out = self.Decoder(encoder_out, decoder_input, look_ahead_mask, encoder_decoder_mask)
        
        return decoder_out
        
    def padding_mask(self, q, k) :
        '''
        q shape : [batch, seq_len]
        k shape : [batch, seq_len]
        ''' 
        q_seq_len = q.size()[-1]
        k_seq_len = k.size()[-1]
        
        q = q.ne(self.padding_idx) # 패딩 토큰은 0, 이외의 토큰은 1
        q = q.unsqueeze(1).unsqueeze(3) # output shape : [batch, 1, q_seq_len, 1]
        q = q.repeat(1, 1, 1, k_seq_len) # output shape : [batch, 1, q_seq_len, k_seq_len]
        
        k = k.ne(self.padding_idx)
        k = k.unsqueeze(1).unsqueeze(2) # output shape : [batch, 1, 1, k_seq_len]
        k = k.repeat(1, 1, q_seq_len, 1) # output shape : [batch, 1, q_seq_len, k_seq_len]
        
        mask = q & k # output shape : [batch, 1, q_seq_len, k_seq_len]
        
        return mask
    
    def look_ahead_mask(self, decoder_input) :
        seq_len = decoder_input.size()[-1]
        
        mask  = torch.tril(torch.ones(seq_len, seq_len)).type(torch.BoolTensor).to(self.device) # output shape : [seq_len, seq_len]
        
        return mask