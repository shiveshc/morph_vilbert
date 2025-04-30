import numpy as np
import torch
import torch.nn as nn

from typing import Tuple, Dict

class Attention(nn.Module):
    def __init__(self,
                 model_dim:int,
                 head_dim:int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.head_dim = head_dim

        self.wq = nn.Linear(model_dim, head_dim)
        self.wk = nn.Linear(model_dim, head_dim)
        self.wv = nn.Linear(model_dim, head_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,
                q:torch.Tensor,
                k:torch.Tensor,
                v:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        att = self.softmax(torch.einsum('bij,bkj->bik', q, k)/np.sqrt(self.head_dim))
        out = torch.einsum('bij,bjk->bik', att, v)
        return att, out


class SelfAttention(Attention):
    def __init__(self,
                 model_dim:int,
                 head_dim:int) -> None:
        super().__init__(model_dim, head_dim)
    
    def forward(self,
                x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x, x, x)


class MHSelfAttention(nn.Module):
    def __init__(self,
                 model_dim:int,
                 num_heads:int) -> None:
        super().__init__()
        head_dim = model_dim//num_heads

        self.heads = [SelfAttention(model_dim, head_dim) for n in range(num_heads)]
        self.ow = nn.Linear(model_dim, model_dim)
    
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = [head(x) for head in self.heads]
        att = torch.stack([o[0] for o in out], dim=3)
        out = torch.concatenate([o[1] for o in out], dim=-1)
        
        out = self.ow(out)
        return att, out


class MLP(nn.Module):
    def __init__(self,
                 model_dim:int,
                 hidden_dim:int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(model_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, model_dim)
        self.act = nn.ReLU()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.act(self.layer1(x))
        x = self.layer2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self,
                 model_dim:int,
                 num_heads:int,
                 mlp_inner_dim:int,
                 **kwargs) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.mha = MHSelfAttention(model_dim, num_heads)
        self.mlp = MLP(model_dim, mlp_inner_dim)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        att, y = self.mha(x)
        x += self.dropout1(y)

        x = self.norm2(x)
        x = x + self.dropout2(self.mlp(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 config:Dict) -> None:
        super().__init__()
        self.config = config

        # layers = [TransformerLayer(config['model']['model_dim'],
        #                            config['model']['num_heads'],
        #                            config['model']['mlp_inner_dim']) for n in range(config['model']['num_layers'])]
        layers = [nn.TransformerEncoderLayer(config['model']['model_dim'],
                                             config['model']['num_heads'],
                                             config['model']['mlp_inner_dim']) for n in range(config['model']['num_layers'])]
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x

        

