import spacy
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

''' Defined User Function '''
# Make Vocab Function
def build_vocab(data_iter, language) :
    language_index = {SRC_LANGUAGE : 0, TRG_LANGUAGE : 1}
    def yield_tokens(data_iter) :
        for data in data_iter :
            yield tokenizer_src(data[language_index[language]]) if language_index[language] == 0 else tokenizer_trg(data[language_index[language]])
    return build_vocab_from_iterator(yield_tokens(data_iter), min_freq=2, specials=["<unk>", "<sos>", "<eos>"], special_first=True)

# Padding
def pad_sequences(sentences, pad_value = 0) :
    max_len = max([len(sentence) for sentence in sentences])
    # padding
    padded_sequences = [sentences + [pad_value] * (max_len - len(sentences)) for sentences in sentences]
    return padded_sequences

# Tokenizing Function
def tokenizing(dataset) :
    enc_input, dec_input, dec_target = [], [], []
    print("Making tokenizing.")
    for data in tqdm(dataset) :
        src_data, trg_data = data
        input, output = [], [vocab_trg["<sos>"]]
        # make input
        for word in src_data :
            try :
                input.append(vocab_src[word.lower()])
            except :
                input.append(vocab_src["<unk>"])
        # make output & target
        for word in trg_data :
            try :
                output.append(vocab_trg[word.lower()])
            except :
                output.append(vocab_trg["<unk>"])
        target = output[1:] + [vocab_trg["<eos>"]]
        
        enc_input.append(input)
        dec_input.append(output)
        dec_target.append(target)
    # padding
    padded_enc_input = pad_sequences(enc_input)
    padded_dec_input = pad_sequences(dec_input)
    padded_dec_target = pad_sequences(dec_input)
    # return
    return [(enc_inp, dec_inp, dec_tar) for enc_inp, dec_inp, dec_tar in zip(padded_enc_input, padded_dec_input, padded_dec_target)]

''' Model Class '''
class Encoder(nn.Module) :
    def __init__(self, src_vocab_size, embedding_dim, hidden_units) :
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx =0)
        self.lstm = nn.LSTM(embedding_dim, hidden_units, batch_first = True)
        
    def forward(self, x) :
        x = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Decoder(nn.Module) :
    def __init__(self, trg_vocab_size, embedding_dim, hidden_units) :
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(trg_vocab_size, embedding_dim, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_dim + hidden_units, hidden_units, batch_first = True)
        self.fc = nn.Linear(hidden_units, trg_vocab_size)
        self.softmax = nn.Softmax(dim = 2)

    def forward(self, x, enc_output, hidden, cell) :
        x = self.embedding(x)
        
        # Dot product attention
        # enc_output.shape : (batch_size, seq_len, hidden_units)
        # hidden.shape : (1, batch_size, hidden_units)
        # hidden.transpose(0, 1).transpose(1, 2).shape : (batch_size, hidden_units, 1)
        # attention_scores.shape : (batch_size, seq_len, 1)
        attention_score = torch.bmm(enc_output, hidden.transpose(0, 1).transpose(1, 2))
        
        # attention_weights.shape : (batch_size, source_seq_len, 1)
        attention_weights = self.softmax(attention_score)
        
        # context_vector.shape = (batch_size, 1, hidden_unit)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), enc_output)
        
        seq_len = x.shape[1]
        context_vector_repeatd = context_vector.repeat(1, seq_len, 1)
        
        # Concatenate context vector and embedded input
        # x.shape : (batch_size, target_seq_len, embedding_dim + hidden_unit)
        x = torch.cat((x, context_vector_repeatd), dim = 2)
        
        # output.shape : (batch_size, target_seq_len, hidden_unit)
        # hidden.shape : (1, batch_size, hidden_unit)
        # cell.shape : (1, batch_size, hidden_unit)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        
        # outputshape : (batch_size, target_seq_len, tar_vocab_size)
        output = self.fc(output)
        ### 
        
        return output, hidden, cell

class Seq2Seq(nn.Module) :
    def __init__(self, encoder, decoder) :
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg) :
        enc_output, hidden, cell = self.encoder(src)
        output, _, _ = self.decoder(trg, enc_output, hidden, cell)
        return output
        
def evaluation(model, dataloader, loss, device) :
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    
    with torch.no_grad() :
        for encoder_inputs, decoder_inputs, decoder_targets in dataloader :
            encoder_inputs = encoder_inputs.to(device)
            decoder_inputs = decoder_inputs.to(device)
            decoder_targets = decoder_targets.to(device)
            
            outputs = model(encoder_inputs, decoder_inputs)
            
            loss = loss_function(outputs.view(-1, outputs.size(-1)), decoder_targets.view(-1))
            total_loss += loss.item()
            
            # 정확도 계산
            mask = decoder_targets != 0
            total_correct += ((outputs.argmax(dim = -1) == decoder_targets) * mask).sum().item()
            total_count += mask.sum().item()
            
        return total_loss / len(dataloader), total_correct / total_count
            
''' Main '''
if __name__ == "__main__" :
    # Setting Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # GPU for MAC

    # Define tokenizer
    SRC_LANGUAGE, TRG_LANGUAGE = "de", "en"
    tokenizer_src = get_tokenizer("spacy", language = "de_core_news_sm")
    tokenizer_trg = get_tokenizer("spacy", language = "en_core_web_sm")

    # Load Dataset
    dataset = Multi30k(split = "train", language_pair = (SRC_LANGUAGE, TRG_LANGUAGE))
    dataset = list(dataset)[:3000]

    # Make Vocab
    vocab_src = build_vocab(dataset, SRC_LANGUAGE)
    vocab_trg = build_vocab(dataset, TRG_LANGUAGE)
    src_vocab_size = len(vocab_src)
    trg_vocab_size = len(vocab_trg)

    # tokenizing process
    tokenized_dataset = tokenizing(dataset)

    # data split
    train_data, valid_data = train_test_split(tokenized_dataset, test_size = 0.2, random_state = 42)

    # Make DalaLoader
    BATCH_SIZE = 64
    encoder_input_train, decoder_input_train, decoder_target_train = list(zip(*train_data))
    encoder_input_valid, decoder_input_valid, decoder_target_valid = list(zip(*valid_data))
    
    encoder_input_train = torch.tensor(encoder_input_train, dtype = torch.long)
    decoder_input_train = torch.tensor(decoder_input_train, dtype = torch.long)
    decoder_target_train = torch.tensor(decoder_target_train, dtype = torch.long)

    encoder_input_valid = torch.tensor(encoder_input_valid, dtype = torch.long)
    decoder_input_valid = torch.tensor(decoder_input_valid, dtype = torch.long)
    decoder_target_valid = torch.tensor(decoder_target_valid, dtype = torch.long)

    train_dataset = TensorDataset(encoder_input_train, decoder_input_train, decoder_target_train)
    valid_dataset = TensorDataset(encoder_input_valid, decoder_input_valid, decoder_target_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle = False)
    
    embedding_dim = 256
    hidden_units = 256

    encoder = Encoder(src_vocab_size, embedding_dim, hidden_units)
    decoder = Decoder(trg_vocab_size, embedding_dim, hidden_units)
    model = Seq2Seq(encoder, decoder)

    loss_function = nn.CrossEntropyLoss(ignore_index = 0)
    optimizer = optim.Adam(model.parameters())

    epochs = 5
    model.to(device)

    # Training loop
    best_valid_loss = float('inf')

    for e in range(epochs) :
        model.train()
        
        for encoder_inputs, decoder_inputs, decoder_targets in tqdm(train_dataloader) :
            encoder_inputs = encoder_inputs.to(device)
            decoder_inputs = decoder_inputs.to(device)
            decoder_targets = decoder_targets.to(device)
            
            outputs = model(encoder_inputs, decoder_inputs)
            
            optimizer.zero_grad()
            loss = loss_function(outputs.view(-1, outputs.size(-1)), decoder_targets.view(-1))
            loss.backward()
            
            # 가중치 업데이트
            optimizer.step()
            
        train_loss, train_acc = evaluation(model, train_dataloader, loss_function, device)
        valid_loss, valid_acc = evaluation(model, valid_dataloader, loss_function, device)
            
        print(f'Epoch : {e+1}/{epochs} | Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.4f} | Valid Loss : {valid_loss:.4f} | Valid Acc : {valid_acc:.4f}')
        
        # Save Checkpoint
        if valid_loss < best_valid_loss :
            print(f'Valiation loss improved from {best_valid_loss:.4f} to {valid_loss:.4f}')
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model_checkpoint.pth')
