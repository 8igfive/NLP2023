import pdb
import torch
from torch import nn
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

        # if torch.cuda.is_available():
        #     self.use_cuda = True
        # else:
        #     self.use_cuda = False
        self.use_cuda = False
        _lambda_params = [torch.tensor([1e-3 * 1, 1e-4 * 1, 1e-5 * 1]),  
                          torch.tensor([1e-3 * 2, 1e-4 * 2, 1e-5 * 2]), 
                          torch.tensor([1e-3 * 3, 1e-4 * 3, 1e-5 * 3]), 
                          torch.tensor([1e-3 * 3, 1e-4 * 3, 1e-5 * 3])]
        # _lambda_params = [torch.tensor([1e-3 * i, 5e-4 * i, 1e-5 * i]) for i in range(1, 5)]
        self.lambda_params = nn.ParameterList([nn.Parameter(tensor) for tensor in _lambda_params])
        self.f0 = nn.Parameter(torch.tensor(1e-7, device=torch.device("cuda" if self.use_cuda else "cpu")))


    def fit(self, corpus: List[str]):

        print(f"\nFitting corpus of size {len(corpus)}...")

        for i in range(len(corpus)):
            unitoken = corpus[i]
            self.unigram_count[unitoken] = self.unigram_count.get(unitoken, 0) + 1
            if i + 1 < len(corpus):
                bitoken = tuple(corpus[i: i + 2])
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

    def initial_p(self, bitoken: Tuple[str, str]) -> float:
        return self.bigram_count.get(bitoken, 1) / self.bigram_count_sum

    def f_3(self, tritokens: List[Tuple[str, str, str]]) -> torch.Tensor:
        f = torch.empty(len(tritokens), dtype=torch.float32, device=torch.device("cuda" if self.use_cuda else "cpu"))
        for i, tritoken in enumerate(tritokens):
            if tritoken[:2] not in self.trigram_freq or \
                tritoken[2] not in self.trigram_freq[tritoken[:2]]:
                f[i] = 1 / self.bigram_count_sum
            else:
                f[i] = self.trigram_freq[tritoken[:2]][tritoken[2]] / self.trigram_freq_sum[tritoken[:2]]
        return f

    def forward(self, tritokens: List[Tuple[str, str, str]]) -> torch.Tensor:
        f_0 = torch.stack([self.f0 for _ in range(len(tritokens))])
        f_1 = torch.empty(len(tritokens), dtype=torch.float32, device=torch.device("cuda" if self.use_cuda else "cpu"))
        f_2 = torch.empty(len(tritokens), dtype=torch.float32, device=torch.device("cuda" if self.use_cuda else "cpu"))
        f_3 = torch.empty(len(tritokens), dtype=torch.float32, device=torch.device("cuda" if self.use_cuda else "cpu"))
        counts = torch.empty(len(tritokens), 3, dtype=torch.float32, device=torch.device("cuda" if self.use_cuda else "cpu"))
        counts[:, 2] = 1
        for i, tritoken in enumerate(tritokens):
            counts[i][0] = self.bigram_count.get(tritoken[:2], 1)
            counts[i][1] = self.unigram_count.get(tritoken[0], 1)
            
            f_1[i] = self.unigram_count.get(tritoken[2], 1) / self.unigram_count_sum
            
            if tritoken[1] not in self.bigram_freq or \
                tritoken[2] not in self.bigram_freq[tritoken[1]]:
                f_2[i] = 1 / self.bigram_count_sum
            else:
                f_2[i] = self.bigram_freq[tritoken[1]][tritoken[2]] / self.bigram_freq_sum[tritoken[1]]
            
            if tritoken[:2] not in self.trigram_freq or \
                tritoken[2] not in self.trigram_freq[tritoken[:2]]:
                f_3[i] = 1 / self.bigram_count_sum
            else:
                f_3[i] = self.trigram_freq[tritoken[:2]][tritoken[2]] / self.trigram_freq_sum[tritoken[:2]]
        lambdas = [counts @ lambda_param for lambda_param in self.lambda_params] # List[tensor of shape(bs)]
        fs = torch.stack([f_0, f_1, f_2, f_3], dim=-1)
        lambdas =  torch.softmax(torch.stack(lambdas, dim=-1), dim=-1)
        lm_p = (fs * lambdas).sum(dim=-1)
        # pdb.set_trace()
        return lm_p
    
    def save_model(self, save_path: str):
        torch.save({
            "unigram_count": self.unigram_count,
            "unigram_count_sum": self.unigram_count_sum,
            "bigram_count": self.bigram_count,
            "bigram_count_sum": self.bigram_count_sum,
            "bigram_freq": self.bigram_freq,
            "bigram_freq_sum": self.bigram_freq_sum,
            "trigram_freq": self.trigram_freq,
            "trigram_freq_sum": self.trigram_freq_sum,
            "state_dict": self.state_dict()
        }, save_path)
    
    def load_model(self, model_path: str):
        ckpt = torch.load(model_path)
        self.unigram_count = ckpt["unigram_count"]
        self.unigram_count_sum = ckpt["unigram_count_sum"]
        self.bigram_count = ckpt["bigram_count"]
        self.bigram_count_sum = ckpt["bigram_count_sum"]
        self.bigram_freq = ckpt["bigram_freq"]
        self.bigram_freq_sum = ckpt["bigram_freq_sum"]
        self.trigram_freq = ckpt["trigram_freq"]
        self.trigram_freq_sum = ckpt["trigram_freq_sum"]
        self.load_state_dict(ckpt["state_dict"])
