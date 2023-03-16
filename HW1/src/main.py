import os
import pdb
import torch
import random
from typing import List, Tuple
from preprocess import PreProcess
from model import LanguageModel
from train import Trainer

def set_random_seed(seed: int):

    print("\n[MAIN] Setting random seed...")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_data(test_file_index: int) -> Tuple[Tuple[List[str], List[str]], List[str]]:

    print("\n[MAIN] Loading data...")

    docs_dir = r"../resources/jyxstxtqj_downcc.com"
    catalog_file = os.path.join(docs_dir, "inf.txt")
    with open(catalog_file, 'r', encoding="gb2312") as fi:
        docs_name = list(map(lambda x: f"{x}.txt", fi.read().strip().split(',')))
    test_files = [os.path.join(docs_dir, docs_name[test_file_index])]
    train_files = [os.path.join(docs_dir, doc_name) for doc_name in docs_name if doc_name not in test_files]
    pprocess = PreProcess(train_files, test_files)
    return pprocess.process()

def build_models(train_corpus_primary: List[str],
                 train_corpus_heldout: List[str]) -> Tuple[LanguageModel, LanguageModel]:
    
    print("\n[MAIN] Building models...")

    model_train = LanguageModel()
    model_train.fit(train_corpus_primary)

    model_target = LanguageModel()
    model_target.fit(train_corpus_heldout)

    return model_train, model_target

def train_model(model_train: LanguageModel, model_target: LanguageModel,
                train_corpus: List[str]):
    
    print("\n[MAIN] Training models...")

    lr = 1e-4
    epochs = 2
    batch_size = 2**17
    print_interval = 1

    
    trainer = Trainer(model_train, model_target, train_corpus, lr)

    trainer.train(epochs, batch_size, print_interval)


def calc_cross_entropy():
    pass

if __name__ == "__main__":
    set_random_seed(0)
    train_corpus, test_corpus = load_data(0)
    model_train, model_target = build_models(*train_corpus)
    train_model(model_train, model_target, train_corpus[1])

    pdb.set_trace()