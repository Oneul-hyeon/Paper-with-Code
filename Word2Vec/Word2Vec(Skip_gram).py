import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
from torch import nn

def SkipGram_pair(window_size, word_sequence) :
    pair = []
    for sequence in word_sequence :
        for i in range(len(sequence)) :
            context = word2index[sequence[i]]
            target = []
            for j in range(window_size, -1, -1) :
                if i - j >= 0 : target.append(word2index[sequence[i-j]]) # 과거 문자 삽입
            for j in range(1, window_size + 1) :
                if i + j < len(sequence) : target.append(word2index[sequence[i+j]] ) # 미래 문자 삽입
            # 페어 생성
            for t in target : pair.append([context, t])

    x_train, y_train = list(zip(*pair))
    return torch.LongTensor(list(x_train)), torch.LongTensor(list(y_train))

class SkipGram(nn.Module) :

    def __init__(self, vocab_size, dimention_size) :
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, dimention_size)
        self.linear = nn.Linear(dimention_size, vocab_size, bias = False)
        self.activation = nn.LogSoftmax(dim = 1)

    def forward(self, X) :
        X = self.embeddings(X)
        X = self.linear(X)
        X = self.activation(X)
        return X

def find_similarity(target_word) :
    target_word_embed = model.state_dict()['embeddings.weight'][word2index[target_word]]

    similarity = []
    for i in range(len(word2index)):
        if target_word != index2word[i]:
            similarity.append(( i, F.cosine_similarity(target_word_embed.unsqueeze(0), model.state_dict()['embeddings.weight'][i].unsqueeze(0)).item()))
        else:
            similarity.append((i, -1)) # target_word와 동일 단어는 -1 처리

    # 유사도 내림차순 정렬
    similarity.sort(key = lambda x : -x[1])

    # 인덱스를 단어로 변환
    print(f'{target_word}와 유사한 단어:')
    for i in range(3) :
        print(f'{i+1}위 : {index2word[similarity[i][0]]}({similarity[i][1]})')

if __name__ == "__main__" :
    # 데이터 준비
    data = [
        'drink cold milk',
        'drink cold water',
        'drink cold cola',
        'drink sweet juice',
        'drink sweet cola',
        'eat delicious bacon',
        'eat sweet mango',
        'eat delicious cherry',
        'eat sweet apple',
        'juice with sugar',
        'cola with sugar',
        'mango is fruit',
        'apple is fruit',
        'cherry is fruit',
        'Berlin is Germany',
        'Boston is USA',
        'Mercedes from Germany',
        'Mercedes is car',
        'Ford from USA',
        'Ford is car'
    ]

    word_sequence = [sequence.split() for sequence in data]
    word_list = list(set(" ".join(data).split()))
    word2index = {key : idx for idx, key in enumerate(word_list) if len(key) > 1}
    index2word = {value : key for key, value in word2index.items()}
    vocab_size = len(word_list)

    window_size = 2
    batch_size = 10

    # Dataset 생성
    x_train, y_train = SkipGram_pair(window_size, word_sequence)
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    dimension_size = 10
    epochs = 1000

    # Training
    model = SkipGram(vocab_size, dimension_size)
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()

    for i in range(epochs+1) :
        for feature, label in train_dataloader :
            out = model(feature)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 100 == 0 : print("epoch : {:d}, loss : {:0.3f}".format(i, loss))

    # 유사도 상위 3개 단어 추출
    find_similarity("cola")