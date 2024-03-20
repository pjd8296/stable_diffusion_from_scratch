import torch
import torch.nn as nn
import torch.nn.functional as F
from sd.attention import SelfAttention


class CLIPEmbedding(nn.Module):
    
    def __init__(self, n_vocab: int, n_embed: int, n_token: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.pos_embedding = nn.Parameter(torch.zeros(n_token, n_embed))
    
    def forward(self, token):
        
        # (batch, seq_len) -> (batch, seq_len, dim)
        x = self.token_embedding(token)
        x += self.pos_embedding
        
        return x

class CLIPLayer(nn.Module):
    
    def __init__(self, num_head: int, n_embed: int):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(num_head, n_embed)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # (batch, seq_len, dim)
        residue = x
        
        # Self-Attention
        x = self.ln_1(x)
        x = self.attention(x, causal_mask=True)
        
        x += residue
        
        # feed-forward connection
        residue = x
        
        x = self.ln_2(x)
        x = self.linear_1(x)
        
        x = x * torch.sigmoid(1.762 * x)    # Quick GeLU activation
        x = self.linear_2(x)
        
        x += residue
        
        return x


class CLIP(nn.Module):
    
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)
        
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])
        
        self.ln = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        
        tokens = tokens.type(torch.long)
        
        # (batch, seq_len) -> (batch, seq_len, dim)
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        
        # (batch, seq_len, dim)
        output = self.ln(state)
        
        return output