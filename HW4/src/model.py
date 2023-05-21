import torch
from torch import nn
from torch.nn import functional as F

class LSTMBlock(nn.Module):
    def __init__(self, embed_size: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(embed_size, embed_size, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout)
        )
        self.l1 = nn.LayerNorm(embed_size)
        self.l2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x_res = x
        x, _ = self.lstm(self.l1(x))
        x = x_res + self.lstm_dropout(x)
        x = x + self.feedforward(self.l2(x))
        
        return x
    
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, layer_num: int, dropout: float = 0.2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.blocks = nn.Sequential(*[LSTMBlock(embed_size, dropout) for _ in range(layer_num)])
        self.lf = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)
        x = self.blocks(x)
        x = self.lf(x)
        
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, _ = logits.shape
            logits = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss