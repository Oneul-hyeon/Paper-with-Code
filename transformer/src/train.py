import os
import pandas as pd
import torch
from tqdm import tqdm
from tokenizer import GetTokenizer
from sklearn.model_selection import train_test_split
from builder import TranslationDataset, TranslationDataLoader
from transformer import Transformer
import torch.nn as nn
import torch.optim as optim
from scheduler import Scheduler
import logging
import time
import math
import matplotlib.pyplot as plt

def Train() :
    train_loss = 0
    model.train()
    for idx, (source, target_input, target_output) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        source = source.to(device)
        target_input = target_input.to(device)
        target_output = target_output.to(device) # shape : [batch, seq_len]
        
        optimizer.zero_grad()

        output = model(source, target_input) # output shape : [batch, seq_len, vocab_size]
        output = output.contiguous().view(-1, output.size()[-1])
        target_output = target_output.contiguous().view(-1)

        loss = loss_fn(output, target_output)
        loss.backward()
        scheduler.step()
        train_loss += loss.item()

    return train_loss / (idx + 1)

def Validation() :
    valid_loss = 0
    model.eval()
    with torch.no_grad() :
        for idx, (source, target_input, target_output) in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            source = source.to(device)
            target_input = target_input.to(device)
            target_output = target_output.to(device) # shape : [batch, seq_len]

            output = model(source, target_input) # output shape : [batch, seq_len, vocab_size]
            output = output.contiguous().view(-1, output.size()[-1])
            target_output = target_output.contiguous().view(-1)
            
            loss = loss_fn(output, target_output)
            valid_loss += loss.item()

    return valid_loss / (idx + 1)

def calculate_time(start, end) :
    total_seconds = end - start
    minute, seconds = total_seconds // 60, total_seconds % 60
    hour, minute = minute // 60, minute % 60
    return hour, minute, seconds

def save_history(epochs, train_loss, valid_loss) :
    epochs = range(1, epochs+1)
    
    plt.figure(figsize=(10,8))
    
    plt.plot(epochs, train_loss, label="train loss", marker="o", linestyle="-")
    plt.plot(epochs, valid_loss, label="valid loss", marker="x", linestyle="-")
    
    plt.title("Train and Validation Loss")
    plt.xlabel("epochs")
    plt.ylabel("value")
    plt.legend()
    
    plt.savefig(PLOT_PATH)
    
if __name__=="__main__" :
    # logging
    logging.basicConfig(filename="log/training.log",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
    
    NOW_DIR=os.path.dirname(__file__)
    ROOT_PATH=os.path.abspath(os.path.join(NOW_DIR, os.pardir))
    DF_PATH=os.path.join(ROOT_PATH, "data/df/df.csv")
    MODEL_PATH = os.path.join(ROOT_PATH, "model/model.pt")
    PLOT_PATH = os.path.join(ROOT_PATH, "img/loss_history.png")
    
    df=pd.read_csv(DF_PATH)
    source, target = df["source"].to_list(), df["target"].to_list()
    
    sos_token, eos_token="<s>", "</s>"
    target_input = [sos_token + sequence for sequence in target]
    target_output = [sequence + eos_token for sequence in target]
    
    # tokenizing
    tokenizer = GetTokenizer()
    tokenized_source = tokenizer.kr_encode(source)
    tokenized_target_input = tokenizer.en_encode(target_input)
    tokenized_target_output = tokenizer.en_encode(target_output)
    
    # make tokenized df 
    tokenized_df=pd.DataFrame({"source" : tokenized_source, "target_input" : tokenized_target_input, "target_output" : tokenized_target_output})
    # split
    train_df, valid_df = train_test_split(tokenized_df, test_size=0.15, random_state=42)
    print(f"train_df shape : {train_df.shape} | valid_df shape : {valid_df.shape}")
    logging.info(f"train_df shape : {train_df.shape} | valid_df shape : {valid_df.shape}")
    
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    # hyperparameter setting
    batch_size = 64
    src_vocab_size = tokenizer.kr_vocab_size()
    tgt_vocab_size = tokenizer.en_vocab_size()
    d_model = 512
    src_max_len = len(tokenized_source[0])
    tgt_max_len = len(tokenized_target_input[0])
    n = 6
    head = 8
    dropout = 0.1
    d_ff = 2048
    padding_idx = tokenizer.get_padding_idx()
    warmup_steps=4000
    
    # make dataset & dataloader
    train_dataset = TranslationDataset(train_df["source"], train_df["target_input"], train_df["target_output"])
    valid_dataset = TranslationDataset(valid_df["source"], valid_df["target_input"], valid_df["target_output"])
    
    train_dataloader = TranslationDataLoader(data=train_dataset, batch_size=batch_size)
    valid_dataloader = TranslationDataLoader(data=valid_dataset, batch_size=batch_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Transformer(src_vocab_size=src_vocab_size,
                    tgt_vocab_size=tgt_vocab_size,
                    d_model=d_model,
                    src_max_len=src_max_len,
                    tgt_max_len=tgt_max_len,
                    n=n,
                    head=head,
                    dropout=dropout,
                    d_ff=d_ff,
                    padding_idx=padding_idx,
                    device=device).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = Scheduler(optimizer=optimizer,
                          d_model=d_model,
                          warmup_steps=warmup_steps)
    
    epochs = 5
    logging.info("training start...")
    best_loss = float("INF")
    train_loss_history, valid_loss_history = [], []
    total_start_time = time.time()
    for epoch in range(1, epochs+1) :
        start_time = time.time()
        
        train_loss = Train() # training 진행
        valid_loss = Validation() # validation 진행
        
        end_time = time.time()
        hour, minute, second = calculate_time(start_time, end_time)
        # 더 작은 loss 값을 가지는 경우 모델 save
        if valid_loss < best_loss :
            best_loss = valid_loss
            torch.save(model.state_dict(), MODEL_PATH)
        
        print(f"Epoch : {epoch} / {epochs} | Time : {hour}h {minute}m {second:.2f}s")
        logging.info(f"Epoch : {epoch} / {epochs} | Time : {hour}h {minute}m {second:.2f}s")
        
        print(f"\tTrain Loss : {train_loss:.3f} | Train PPL : {math.exp(train_loss):.3f}")
        logging.info(f"\tTrain Loss : {train_loss:.3f} | Train PPL : {math.exp(train_loss):.3f}")
        
        print(f"\tValid Loss : {valid_loss:.3f} | Valid PPL : {math.exp(valid_loss):.3f}\n")
        logging.info(f"\tValid Loss : {valid_loss:.3f} | Train PPL : {math.exp(valid_loss):.3f}\n")
    
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
    
    total_end_time = time.time()
    hour, minute, seconds = calculate_time(total_start_time, total_end_time)
    logging.info(f"Total Training Time : {hour}h {minute}m {seconds:.2f}s")
    save_history(epochs, train_loss_history, valid_loss_history)