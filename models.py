import torch
import torch.nn as nn
from mamba_ssm import Mamba

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: str ='cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class MambaBlock(nn.Module):
    def __init__(self, embed_dim, state_size=64, d_conv=3, expand=2):
        super().__init__()

        self.mamba =  Mamba(d_model=embed_dim, d_state=state_size, d_conv=d_conv, expand=expand)
        self.norm = RMSNorm(embed_dim)

    def forward(self, x):
        x = self.norm(self.mamba(x))
        return x

class gloveEmbedding(nn.Module):
    def __init__(self, device='cuda'):
        self.device = device
    def __call__(self, x):
        return x.to(self.device)
    
class protvecEmbedding(nn.Module):
    def __init__(self, device='cuda'):
        self.device = device
    def __call__(self, x):
        x = x.view(x.size(0), 3, 100)
        return x.to(self.device)

class MambaTower(nn.Module):
    def __init__(self, embed_dim, n_layers, state_size, dropout=0.3, global_pool=True, d_conv=3, expand=2, embedding = 'learned', device='cuda'):
        super().__init__()

        self.__name__ = 'mambatower'
        if embedding == 'learned':
            self.embedding_layer = nn.Embedding(num_embeddings=21, embedding_dim=embed_dim)
        if embedding == 'glove':
            self.embedding_layer = gloveEmbedding(device)
        if embedding == 'protvec':
            self.embedding_layer = protvecEmbedding(device)
        self.blocks = nn.Sequential(*[MambaBlock(embed_dim, state_size, d_conv, expand) for _ in range(n_layers)])
        self.global_pool = global_pool
        self.fc = nn.Linear(embed_dim, 2)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.embedding_layer(x.type(torch.long)).type(torch.float32)
        out = self.blocks(out) if not self.global_pool else torch.mean(self.blocks(out),1)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out
    
class Transformer(nn.Module):
    def __init__(self, embed_dim, n_head, n_layers, state_size, dropout=0.3, global_pool=True, expand=2):
        super().__init__()
        self.__name__ = 'transformer'
        silu = nn.SiLU()
        self.embedding_layer = nn.Embedding(num_embeddings=21, embedding_dim=embed_dim)
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, dim_feedforward=state_size*expand , dropout=0.0, activation=silu, batch_first=True)
        self.norm = RMSNorm(embed_dim)        
        self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=n_layers, norm=self.norm)
        self.global_pool = global_pool
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        out = self.embedding_layer(x.type(torch.long)).type(torch.float32)
        out = self.encoder(x) if not self.global_pool else torch.mean(self.encoder(out),1)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
class RNN(nn.Module):
        def __init__(self, embed_dim, hidden_size, num_layers, dropout=0.3, global_pool=True, expand=2):
            super().__init__()
            self.__name__ = 'RNN'
            self.embedding_layer = nn.Embedding(num_embeddings=21, embedding_dim=embed_dim)
            self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,bidirectional=False)
            self.norm = RMSNorm(hidden_size)
            self.global_pool = global_pool
            self.act = nn.Tanh()
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, expand*hidden_size)
            self.fc2 = nn.Linear(expand*hidden_size, 2)

        def forward(self, x):
            out = self.embedding_layer(x.type(torch.long)).type(torch.float32)
            out, _ = self.rnn(out)
            out = self.norm(out) if not self.global_pool else torch.mean(self.norm(out), 1)
            out = self.act(out)
            out = self.dropout(out)
            out = self.fc1(out)
            out = self.act(out)
            out = self.fc2(out)
            return out
        
class LSTM(nn.Module):
        def __init__(self, embed_dim, hidden_size, num_layers, dropout=0.3, global_pool=True, expand=2):
            super().__init__()
            self.__name__ = 'LSTM'
            self.embedding_layer = nn.Embedding(num_embeddings=21, embedding_dim=embed_dim)
            self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,bidirectional=False)
            self.norm = RMSNorm(hidden_size)
            self.global_pool = global_pool
            self.act = nn.Tanh()
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, expand*hidden_size)
            self.fc2 = nn.Linear(expand*hidden_size, 2)

        def forward(self, x):
            out = self.embedding_layer(x.type(torch.long)).type(torch.float32)
            out, _ = self.rnn(out)
            out = self.norm(out) if not self.global_pool else torch.mean(self.norm(out), 1)
            out = self.act(out)
            out = self.dropout(out)
            out = self.fc1(out)
            out = self.act(out)
            out = self.fc2(out)
            return out