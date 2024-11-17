import torch.nn as nn
import torch
import math

class ScaledDotProductAttention(nn.Module) :
    def __init__(self) :
        super(ScaledDotProductAttention, self).__init__()
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None) :
        '''
        q shape : [batch, head, seq_len, head_dim]
        k shape : [batch, head, seq_len, head_dim]
        v shape : [batch, head, seq_len, head_dim]
        '''
        # 1. 쿼리 행렬과 키 행렬 간의 내적을 계산하고 QK_T 유사도 값을 산출한다.
        k_T = k.transpose(-1, -2) # output shape : [batch, head, head_dim, seq_len]
        attention_score = torch.matmul(q, k_T) # output shape : [batch, head, seq_len, seq_len]
        
        # 2. QK_T를 키 행렬의 차원의 제곱근으로 나눈다.
        d_k = k.size()[-1]
        attention_score /= math.sqrt(d_k) # output shape : [batch, head, seq_len, seq_len]
        
        # 3. 마스킹된 부분이 있다면 -무한으로 값 채우기
        if mask is not None :
            attention_score = attention_score.masked_fill(mask==0, -1e10)
            
        # 4. 스코어 행렬에 softmax 함수를 적용해 정규화 작업을 진행한다.
        attention_score = self.softmax(attention_score) # output shape : [batch, head, seq_len, seq_len]
        
        # 5. 스코어 행렬에 밸류 행렬을 곱해 어텐션 행렬 Z를 산출한다.
        Z = torch.matmul(attention_score, v) # output shape : [batch, head, seq_len, head_dim]
        
        return Z, attention_score
    
class MultiHeadAttention(nn.Module) :
    def __init__(self, d_model, head) :
        super(MultiHeadAttention, self).__init__()
    
        self.d_model = d_model
        self.head = head
        self.head_dim = self.d_model // self.head
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        
    def forward(self, q, k, v, mask=None) :
        '''
        q shape : [batch, seq_len, d_model]
        k shape : [batch, seq_len, d_model]
        v shape : [batch, seq_len, d_model]
        '''
        # 1단계 : q, k, v projection
        q = self.w_q(q) # output shape : [batch, seq_len, d_model]
        k = self.w_k(k) # output shape : [batch, seq_len, d_model]
        v = self.w_v(v) # output shape : [batch, seq_len, d_model]
        
        # 2단계 : Head 수만큼 분할
        batch = q.size()[0]
        q = q.view(batch, -1, self.head, self.head_dim).transpose(1, 2) # output shape : [batch, head, seq_len, d_model]
        k = k.view(batch, -1, self.head, self.head_dim).transpose(1, 2) # output shape : [batch, head, seq_len, d_model]
        v = v.view(batch, -1, self.head, self.head_dim).transpose(1, 2) # output shape : [batch, head, seq_len, d_model]
        
        # 3단계 : Scaled Dot Product Attention 수행
        '''
        Z shape : [batch, head, seq_len, head_dim]
        attention_score shape : [batch, head, seq_len, seq_len]
        '''
        Z, attention_score = self.ScaledDotProductAttention(q, k, v, mask)
        
        # 4단계 : concatenate
        Z = Z.transpose(1, 2).contiguous().view(batch, -1, self.d_model) # output shape : [batch, seq_len, d_model]

        # 5단계 : d_model 차원으로 projection
        Z = self.w_o(Z)
        
        return Z, attention_score