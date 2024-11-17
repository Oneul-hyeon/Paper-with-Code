import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def MakeTokenizer(sequences, vocab_size, language, save_file_path) :
    # tokenizer initializing & setting
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel() if language=="en" else pre_tokenizers.UnicodeScripts()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
        unk_token="<unk>"
    )

    # tokenizer training
    tokenizer.train_from_iterator(sequences, trainer)

    # setting decoder processor
    # tokenizer.decoder = decoders.ByteLevel() if language=="en" else decoders.BPEDecoder()
    tokenizer.decoder = decoders.ByteLevel()
    
    # save tokeniezr
    tokenizer.save(save_file_path)
    
if __name__ == "__main__" :
    NOW_PATH = os.path.dirname(os.path.abspath(__file__))
    ROOT_PATH = os.path.dirname(NOW_PATH)
    DATA_PATH = os.path.join(ROOT_PATH, "data")
    RAW_PATH = os.path.join(DATA_PATH, "raw")
    TOKENIZER_DIR = os.path.join(ROOT_PATH, "tokenizer")
    
    raw_files = [str(path) for path in Path(RAW_PATH).glob("*.xlsx")]
    # collecting
    sequence_kr, sequence_en = [], []
    for raw_file in tqdm(raw_files) :
        df = pd.read_excel(raw_file)
        for _, row in tqdm(df.iterrows(), total=len(df)) :
            context_kr, context_en = row["원문"], row["번역문"]
            if context_kr : sequence_kr.append(context_kr.strip())
            if context_en : sequence_en.append(context_en.strip())

    # make tokenizer
    kr_tokenizer_path = os.path.join(TOKENIZER_DIR, "kr_tokenizer.json")
    en_tokenizer_path = os.path.join(TOKENIZER_DIR, "en_tokenizer.json")

    MakeTokenizer(sequences=sequence_kr,
                  vocab_size=32000,
                  language="kr",
                  save_file_path=kr_tokenizer_path)
    MakeTokenizer(sequences=sequence_en,
                  vocab_size=32000,
                  language="en",
                  save_file_path=en_tokenizer_path)