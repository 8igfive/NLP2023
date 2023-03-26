import os
import pdb
import torch
import random
import multiprocessing
import numpy as np
from typing import List, Tuple
from preprocess import PreProcess
from model import LanguageModel
from train import Trainer

def set_random_seed(seed: int):

    print("\n[MAIN] Setting random seed...")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_data(test_file_index: int, tokenize_word_level: bool = False) -> Tuple[Tuple[Tuple[List[str], List[str]], List[str]], str]:

    print("\n[MAIN] Loading data...")

    docs_dir = r"../resources/jyxstxtqj_downcc.com"
    catalog_file = os.path.join(docs_dir, "inf.txt")
    with open(catalog_file, 'r', encoding="gb2312") as fi:
        docs_name = list(map(lambda x: f"{x}.txt", fi.read().strip().split(',')))
    test_files = [os.path.join(docs_dir, docs_name[test_file_index])]
    train_files = [os.path.join(docs_dir, doc_name) for doc_name in docs_name if os.path.join(docs_dir, doc_name) not in test_files]
    pprocess = PreProcess(train_files, test_files)
    return pprocess.process(tokenize_word_level), docs_name[test_file_index]

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

    lr = 1e-4 # 1e-4
    epochs = 30 # 5
    batch_size = 2**17
    print_interval = 1

    
    trainer = Trainer(model_train, model_target, train_corpus, lr)

    trainer.train(epochs, batch_size, print_interval)


def calc_cross_entropy(model: LanguageModel, corpus: List[str]):
    model.eval()
    log_p = np.log(model.initial_p(tuple(corpus[:2])))
    with torch.no_grad():
        for i in range(len(corpus) - 2):
            log_p +=np.log(model.f_3([tuple(corpus[i: i + 3])]).item())
    return -1 / len(corpus) * log_p

def main(rank: int, tokenize_word_level: bool = False):
    set_random_seed(0)

    cross_entropys = dict()
    for test_file_index in [4 * i + rank for i in range(4)]:
        (train_corpus, test_corpus), test_file = load_data(test_file_index, tokenize_word_level)
        model_train, model_target = build_models(*train_corpus)
        # train_model(model_train, model_target, train_corpus[1])

        cross_entropy = calc_cross_entropy(model_train, test_corpus)
        print(f"\nCross Entropy: {cross_entropy}")
        cross_entropys[test_file] = cross_entropy
    
    return cross_entropys

if __name__ == "__main__":
    cross_entropys = dict()
    tokenize_word_level = True

    with multiprocessing.Pool(4) as pool:
        for ret in pool.starmap(main, [(i, tokenize_word_level) for i in range(4)]):
            cross_entropys.update(ret)

    print(cross_entropys)

    # show = []
    # for i in range(1):
    #     (train_corpus, test_corpus), test_file = load_data(i, True)
    #     show.append((test_file, len(train_corpus[0]) + len(train_corpus[1]), len(test_corpus)))
    # print("\n".join(f"《{os.path.splitext(i)[0]}》\t\t{j}\t\t{k}" for i, j, k in show))
