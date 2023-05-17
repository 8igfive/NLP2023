import torch
import random
from trainer import Corpus, Trainer
from model import LSTMLanguangeModel

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)

    corpus = Corpus(128, 16, 64, 'char', 0.05)
    model = LSTMLanguangeModel(len(corpus.token2id), 384, 12, 0.2)
    trainer = Trainer(model, corpus, 2, (5e-5, 1e-4), 1000, torch.device('cuda:0'))

    trainer.train(100)