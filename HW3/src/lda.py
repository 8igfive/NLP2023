import random
import pdb
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Callable, Union

random.seed(0)

class LDA:
    def __init__(self, alpha: np.ndarray, beta: np.ndarray):
        self.alpha = alpha
        self.beta = beta
        self.K = self.alpha.shape[0]
        self.V = self.beta.shape[0]

    def fit(self, corpus: List[Union[List, int]], max_iter: int = 100, min_topic_change: int = 100):
        # initialization
        W = [para for para, _ in corpus]
        self.M = len(W)
        self.Theta = np.zeros((self.M, self.K), dtype=int)
        self.n_m = np.zeros((self.M,), dtype=int)
        self.Phi = np.zeros((self.K, self.V), dtype=int)
        self.n_k = np.zeros((self.K,), dtype=int)
        Z = []
        for m, w_m in enumerate(W):
            Z.append([])
            for w_mn in w_m:
                z_mn = random.sample(range(self.K), k=1)[0]
                Z[-1].append(z_mn)
                self.Theta[m, z_mn] += 1
                self.n_m[m] += 1
                self.Phi[z_mn, w_mn] += 1
                self.n_k[z_mn] += 1
        
        # iteration
        with tqdm(range(max_iter)) as pbar:
            for _ in pbar:
                topic_change_count = 0
                for m, w_m in enumerate(W):
                    for n, w_mn in enumerate(w_m):
                        z_mn = Z[m][n]
                        self.Theta[m][z_mn] -= 1
                        self.n_m[m] -= 1
                        self.Phi[z_mn][w_mn] -= 1
                        self.n_k[z_mn] -= 1
                        weights = [(self.Phi[k, w_mn].item() + self.beta[w_mn].item()) / (self.n_k[k].item() + self.beta.sum()) * 
                                   (self.Theta[m, k].item() + self.alpha[k].item()) / (self.n_m[m].item() + self.alpha.sum())  
                                   for k in range(self.K)]
                        n_z_mn = random.choices(range(self.K), weights=weights, k=1)[0]
                        Z[m][n] = n_z_mn
                        self.Theta[m][n_z_mn] += 1
                        self.n_m[m] += 1
                        self.Phi[n_z_mn][w_mn] += 1
                        self.n_k[n_z_mn] += 1
                        if n_z_mn != z_mn:
                            topic_change_count += 1
                pbar.set_postfix({'topic_change_count': topic_change_count})
                if topic_change_count <= min_topic_change:
                    break
            
    def get_theta(self):
        if not hasattr(self, 'theta'):
            self.theta = np.empty(self.Theta.shape, dtype=np.float)
            for m in range(self.M):
                for k in range(self.K):
                    self.theta[m,k] = (self.Theta[m, k].item() + self.alpha[k].item()) / (self.n_m[m].item() + self.alpha.sum())
        return self.theta

    def get_phi(self):
        if not hasattr(self, 'phi'):
            self.phi = np.empty(self.Phi.shape, dtype=np.float)
            for k in range(self.K):
                for v in range(self.V):
                    self.phi[k,v] = (self.Phi[k, v].item() + self.beta[v].item()) / (self.n_k[k].item() + self.beta.sum())
        return self.phi