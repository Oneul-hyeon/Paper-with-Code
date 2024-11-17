import torch.nn as nn

class PositionWiseFeedForwardNetwork(nn.Module) :
    def __init__(self, d_model, d_ff) :
        super(PositionWiseFeedForwardNetwork, self).__init__()
        
        self.Linear_1 = nn.Linear(d_model, d_ff)
        self.ReLU = nn.ReLU()
        self.Linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x) :
        '''
        x shape : [batch, seq_len, d_model]
        '''
        x = self.Linear_1(x) # output shape : [batch, seq_len, d_model]
        x = self.ReLU(x) # output shape : [batch, seq_len, d_model]
        x = self.Linear_2(x) # output shape : [batch, seq_len, d_model]

        return x