import os
import torch
from tokenizer import GetTokenizer
from transformer import Transformer

class Translator :
    def __init__(self) :
        NOW_PATH = os.path.dirname(__file__)
        ROOT_PATH = os.path.join(NOW_PATH, os.pardir)
        MODEL_PATH = os.path.join(ROOT_PATH, "model/model_now.pt")
        
        self.device = torch.device("cpu")
        self.tokenizer = GetTokenizer(train=False)
        
        # hyperparameter
        src_vocab_size = self.tokenizer.kr_vocab_size()
        tgt_vocab_size = self.tokenizer.en_vocab_size()
        d_model = 512
        src_max_len = 100
        tgt_max_len = 100
        n = 6
        head = 8
        dropout = 0.1
        d_ff = 2048
        padding_idx = self.tokenizer.get_padding_idx()
    
        self.model = Transformer(src_vocab_size=src_vocab_size,
                                 tgt_vocab_size=tgt_vocab_size,
                                 d_model=d_model,
                                 src_max_len=src_max_len,
                                 tgt_max_len=tgt_max_len,
                                 n=n,
                                 head=head,
                                 dropout=dropout,
                                 d_ff=d_ff,
                                 padding_idx=padding_idx,
                                 device=self.device).to(self.device)

        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        
    def translate(self, sequence) :
        # make encoder_input & decoder_input
        encoder_input = torch.LongTensor(self.tokenizer.kr_encode([sequence])[-1]).unsqueeze(0).to(self.device) # output shape : [batch, seq_len]
        decoder_input = torch.LongTensor(self.tokenizer.en_encode(["<s>"])[-1]).unsqueeze(0).to(self.device) # output.shape : [batch, seq_len]
        
        inference = self.model.inference(encoder_input, decoder_input, self.tokenizer).tolist()
        translation = self.tokenizer.en_decode(inference)
        
        return translation