import torch.nn as nn
import torch

class PositionalEncoding(nn.Module) :
    def __init__(self, max_len, d_model, device) :
        super(PositionalEncoding, self).__init__()
        
        self.max_len = max_len
        self.d_model = d_model
        
        # 빈 텐서 생성
        self.PE = torch.zeros(max_len, d_model, device=device) # output shape = (max_len, d_model)
        self.PE.require_grad = False
        # row 방향으로 인덱싱
        pos = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(dim=1)
        # 2i 구하기
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float, device=device)
        
        # PE 값 설정
        self.PE[:, 0::2] = torch.sin(pos / 10000 ** (_2i//self.d_model))
        self.PE[:, 1::2] = torch.cos(pos / 10000 ** (_2i//self.d_model))
        
        # 치원 추가(input embedding과의 브로드캐스팅을 위해 적용)
        self.PE = self.PE.unsqueeze(0)
    
    def forward(self, x) :
        # 초기 x shape : (batch, seq_len)
        seq_len = x.size()[1]
        # positional encoding 길이 조절 
        return self.PE[:, :seq_len, :]