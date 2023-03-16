import os
import pdb
import time
import torch
import random
from typing import Tuple, List
from torch.optim import Adam
from torch import nn
from model import LanguageModel

class Trainer:
    
    def __init__(self, model_train: LanguageModel, model_target: LanguageModel, 
                 train_corpus: List[str], lr: float = 1e-3):
        self.model_train = model_train
        self.model_target = model_target
        self.train_corpus = train_corpus

        self.optimizer = Adam(model_train.parameters(), lr=lr)

    def train(self, epochs: int, batch_size: int, print_interval: int):
        self.model_train.train()
        self.model_target.eval()
        data_indices = list(range(len(self.train_corpus) - 2))
        loss_fn = nn.MSELoss()
        for epoch in range(epochs):
            print(f"\nEpoch {epoch} starts.\n")
            random.shuffle(data_indices)
            loss_sum = 0
            for index_base in range(0, len(data_indices), batch_size):
                step = index_base // batch_size
                tritokens = []
                for batch_i in range(min(batch_size, len(data_indices) - index_base)):
                    data_index = data_indices[index_base + batch_i]
                    tritokens.append(tuple(self.train_corpus[data_index: data_index + 3]))
                train_p = self.model_train(tritokens)
                target_p = self.model_target.f_3(tritokens)
                loss = loss_fn(train_p, target_p)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
                if step % print_interval == 0:
                    print(f"epoch: {epoch}, step: {step}, loss: {loss.item()}, avg_loss: {loss_sum / (step + 1)}")
        
    def load_models(self, train_path: str, target_path: str):
        self.model_train.load_model(train_path)
        self.model_target.load_model(target_path)

    def save_models(self):
        dump_dir = r'../dump'
        save_name_p = "{} {}.ckpt"
        cur_time = time.asctime()
        os.makedirs(dump_dir, exist_ok=True)
        self.model_train.save_model(os.path.join(dump_dir, save_name_p.format(cur_time, "model_train")))
        self.model_target.save_model(os.path.join(dump_dir, save_name_p.format(cur_time, "model_target")))

