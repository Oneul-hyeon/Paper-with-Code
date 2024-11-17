import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset) :
    def __init__(self, source_input, target_input, target_output) :
        self.source_input = source_input
        self.target_input = target_input
        self.target_output = target_output

    def __len__(self) :
        return len(self.source_input)
    
    def __getitem__(self, idx) :
        return torch.LongTensor(self.source_input[idx]), torch.LongTensor(self.target_input[idx]), torch.LongTensor(self.target_output[idx])

def TranslationDataLoader(data, batch_size, shuffle=True) :
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)