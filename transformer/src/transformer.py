import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module) :
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, src_max_len, tgt_max_len, n, head, dropout, d_ff, padding_idx, device) :
        super(Transformer, self).__init__()
        
        self.padding_idx = padding_idx
        self.device = device
        self.tgt_max_len = tgt_max_len
        
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
    
    def inference(self, encoder_input, decoder_input, tokenizer) :
        encoder_mask = self.padding_mask(encoder_input, encoder_input)
        with torch.no_grad() :
            encoder_out = self.Encoder(encoder_input, encoder_mask)
        
        # 한 토큰씩 생성
        for idx in range(self.tgt_max_len) :
            encoder_decoder_mask = self.padding_mask(decoder_input, encoder_input)
            look_ahead_mask = self.padding_mask(decoder_input, decoder_input) * self.look_ahead_mask(decoder_input)

            with torch.no_grad() :
                # decoder output
                decoder_out = self.Decoder(encoder_out, decoder_input, look_ahead_mask, encoder_decoder_mask) # output shape : [batch, seq_len, vocab_size]
            
            # 가장 높은 확률을 가진 토큰 선택
            next_token_ids = [decoder_out.argmax(2)[:,-1].item()]
            next_token_ids = torch.LongTensor(next_token_ids).unsqueeze(0) # output shape : [batch, 1]
            
            # 예측된 토큰을 decoder input에 추기
            decoder_input = torch.cat([decoder_input, next_token_ids], dim = 1) # output shape : [batch, decoder_out seq_len + 1]
            
            # eos 토큰 반환 시 탈출
            if next_token_ids == tokenizer.en_encode(["</s>"])[-1][-1] :
                break
        
        # 총 토큰 반환
        return decoder_input.squeeze(0)