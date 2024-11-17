import os
from tokenizers import Tokenizer
from tqdm import tqdm

class GetTokenizer() :
    def __init__(self) :
        NOW_DIR = os.path.dirname(__file__)
        ROOT_PATH = os.path.abspath(os.path.join(NOW_DIR, os.pardir))
        TOKENIZER_PATH = os.path.join(ROOT_PATH, "tokenizer")

        TOKENIZER_FILE_KR = os.path.join(TOKENIZER_PATH, "kr_tokenizer.json")
        TOKENIZER_FILE_EN = os.path.join(TOKENIZER_PATH, "en_tokenizer.json")

        self.tokenizer_kr=Tokenizer.from_file(TOKENIZER_FILE_KR)
        self.tokenizer_en=Tokenizer.from_file(TOKENIZER_FILE_EN)

        # padding function
        self.tokenizer_kr.enable_padding(pad_id = self.tokenizer_kr.token_to_id("<pad>"),
                                         pad_token="<pad>",
                                         pad_to_multiple_of=8)
        self.tokenizer_en.enable_padding(pad_id = self.tokenizer_en.token_to_id("<pad>"),
                                         pad_token="<pad>",
                                         pad_to_multiple_of=8)
        
    def kr_encode(self, sequences) :
        encode_batch=self.tokenizer_kr.encode_batch(sequences)
        return [encoding.ids for encoding in encode_batch]
    
    def kr_decode(self, sequence_ids) :
        return self.tokenizer_kr.decode(sequence_ids, skip_special_tokens=False)
    
    def en_encode(self, sequences) :
        encode_batch=self.tokenizer_en.encode_batch(sequences)
        return [encoding.ids for encoding in encode_batch]
    
    def en_decode(self, sequence_ids) :
        return self.tokenizer_en.decode(sequence_ids, skip_special_tokens=False)
    
    def kr_vocab_size(self) :
        return self.tokenizer_kr.get_vocab_size()
    
    def en_vocab_size(self) :
        return self.tokenizer_en.get_vocab_size()
    
    def get_padding_idx(self) :
        return self.tokenizer_kr.token_to_id("<pad>")