import sys
import torch
import random
import logging
from trainer import Corpus, Trainer
from model import LSTMLanguangeModel

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)

    logging.basicConfig(level=logging.INFO)

    corpus = Corpus(128, 4, 256, sys.argv[1], 0.05)
    model = LSTMLanguangeModel(len(corpus.token2id), 384, 12, 0.2)
    trainer = Trainer(model, corpus, 2, (1e-5, 5e-4), 1000, torch.device('cuda:0'))

    trainer.train(100)