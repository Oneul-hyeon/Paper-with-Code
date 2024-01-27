import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
from torch import nn

def CBOW_pair(window_size, word_sequence) :
    x_train, y_train = [], []
    for sequence in word_sequence :
        for i in range(len(sequence)) :
            y_train.append(word2index[sequence[i]])
            context = []
            for j in range(window_size, 0, -1) :
                # 과거 문자 삽입(With padding)
                context.append(word2index[sequence[i-j]] if i - j >= 0 else 0)
            for j in range(1, window_size + 1) :
                # 미래 문자 삽입(With padding)
                context.append(word2index[sequence[i+j]] if i + j < len(sequence) else 0)
            # 페어 생성
            x_train.append(context)
    return torch.LongTensor(x_train), torch.LongTensor(y_train)

class CBOW(nn.Module) :

    def __init__(self, vocab_size, dimension_size) :
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, dimension_size)
        self.linear = nn.Linear(dimension_size, vocab_size, bias = False)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, X) :
        X = self.embeddings(X)
        X = X.sum(dim = 1)
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
        'Ford is a car'
    ]

    word_sequence = [sequence.split() for sequence in data]
    word_list = list(set(" ".join(data).split()))
    word2index = {key : idx for idx, key in enumerate(word_list, start = 1)}
    word2index['<PAD>'] = 0
    index2word = {value : key for key, value in word2index.items()}
    vocab_size = len(word_list) + 1

    window_size = 2
    batch_size = 10

    # Dataset 생성
    x_train, y_train = CBOW_pair(window_size, word_sequence)
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    dimension_size = 5
    epochs = 1000

    # Training
    model = CBOW(vocab_size, dimension_size)
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()

    for i in range(epochs + 1) :
        for feature, label in train_dataloader :
            out = model(feature)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 100 == 0 : print("epoch : {:d}, loss : {:0.3f}".format(i, loss))

    # 유사도 상위 3개 단어 추출
    find_similarity("cola")