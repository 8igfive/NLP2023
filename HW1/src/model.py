import torch
import torch.nn as nn
from typing import Dict, Tuple, List

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unigram_count: Dict[str, int] = dict()
        self.bigram_count: Dict[Tuple[str, str], int] = dict()
        self.bigram_freq: Dict[str, Dict[str, int]] = dict()
        self.trigram_freq: Dict[Tuple[str, str], Dict[str, int]] = dict()

        self.unigram_count_sum: int = 0
        self.bigram_count_sum: int = 0
        self.bigram_freq_sum: Dict[str, int] = dict()
        self.trigram_freq_sum: Dict[Tuple[str, str], int] = dict()
    
        self.lambda_params = nn.ModuleList([nn.Parameter(torch.randn(3)) for _ in range(4)])
        self.f0 = nn.Parameter(torch.randn([]))

    def fit(self, corpus: List[str]):
        for i in range(len(corpus)):
            unitoken = corpus[i]
            self.unigram_count[unitoken] = self.unigram_count.get(unitoken, 0) + 1
            if i + 1 < len(corpus):
                bitoken = corpus[i: i + 2]
                self.bigram_count[bitoken] = self.bigram_count.get(bitoken, 0) + 1
                if bitoken[0] not in self.bigram_freq:
                    self.bigram_freq[bitoken[0]] = dict()
                self.bigram_freq[bitoken[0]][bitoken[1]] = self.bigram_freq[bitoken[0]].get(bitoken[1], 0) + 1
                if i + 2 < len(corpus):
                    tritoken_2 = corpus[i + 2]
                    if bitoken not in self.trigram_freq:
                        self.trigram_freq[bitoken] = dict()
                    self.trigram_freq[bitoken][tritoken_2] = self.trigram_freq[bitoken].get(tritoken_2, 0) + 1
        
        self.unigram_count_sum = sum(self.unigram_count.values())
        self.bigram_count_sum = sum(self.bigram_count.values())
        self.bigram_freq_sum = {key: sum(value.values()) for key, value in self.bigram_freq.items()}
        self.trigram_freq_sum = {key: sum(value.values()) for key, value in self.trigram_freq.items()}

    def initial_p(self, bitoken: Tuple[str, str]):
        return self.bigram_count.get(bitoken, 0) / self.bigram_count_sum

    def forward(self, tritokens: List[Tuple[str, str, str]]):
        f_0 = torch.stack([self.f0 for _ in range(len(tritokens))])
        f_1 = torch.empty(len(tritoken), dtype=torch.float32)
        f_2 = torch.empty(len(tritoken), dtype=torch.float32)
        f_3 = torch.empty(len(tritoken), dtype=torch.float32)
        counts = torch.empty(len(tritoken), 3, dtype=torch.float32)
        counts[:, 2] = 1
        for i, tritoken in enumerate(tritokens):
            counts[i][0] = self.bigram_count.get(tritoken[:2], 0) / self.bigram_count_sum
            counts[i][1] = self.unigram_count.get(tritoken[0], 0) / self.unigram_count_sum
            
            f_1[i] = self.unigram_count.get(tritoken[2], 0) / self.unigram_count_sum
            
            if tritoken[1] not in self.bigram_freq:
                f_2[i] = 0
            else:
                f_2[i] = self.bigram_freq[tritoken[1]].get(tritoken[2], 0) / self.bigram_freq_sum[tritoken[1]]
            
            if tritoken[:2] not in self.trigram_freq:
                f_3[i] = 0
            else:
                f_3[i] = self.trigram_freq[tritoken[:2]].get(tritoken[2], 0) / self.trigram_freq_sum[tritoken[:2]]
        lambdas = [counts @ lambda_param for lambda_param in self.lambda_params] # List[tensor of shape(bs)]
        fs = torch.stack([f_0, f_1, f_2, f_3], dim=-1)
        lambdas =  torch.softmax(torch.stack(lambdas, dim=-1))
        lm_p = (fs * lambdas).sum(dim=-1)
        
        return lm_p