import random
import jieba
import pdb
import re
import os
from tqdm import tqdm
from functools import reduce
from typing import Tuple, List, Callable

BLANK = "[BLK]"
SEPERATION = "[SEP]"

class PreProcess:

    def __init__(self, train_files: List[str], test_files: List[str]):
        print("\nReading files...")
        self.test_docs = []
        for test_file in test_files:
            with open(test_file, 'r', encoding='gb2312', errors='ignore') as fi:
                self.test_docs.append(fi.read())

        print("\nTest documents: {}.".format(list(map(
            lambda x: os.path.splitext(os.path.split(x)[-1])[0],
            test_files
        ))))

        self.train_docs = []
        for train_file in train_files:
            with open(train_file, 'r', encoding='gb2312', errors='ignore') as fi:
                self.train_docs.append(fi.read())
        random.shuffle(self.train_docs)

        print("\nTrain documents: {}.".format(list(map(
            lambda x: os.path.splitext(os.path.split(x)[-1])[0],
            train_files
        ))))

    def _inner_call(self, fn: Callable):
        for name, docs in {
            "Test documents": self.test_docs,
            "Train documents": self.train_docs,
        }.items():
            fn(docs)
            print(f"{name}{'. ' if name == 'Train documents' else ', '}",
                  end=('\n' if name == 'Train documents' else ''))

    def _remove_ad(self):
        print("\nRemoving advertisements for: ", end='')
        ad_p = re.compile(r"本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com")
        def remove(docs: List[str]):
            for i, doc in enumerate(docs):
                docs[i] = ad_p.sub('', doc)
        self._inner_call(remove)

    def _extract_chinese(self):
        print("\nExtracting chinese and common punctuations for: ", end='')
        nc_p = re.compile(r"[^\u4e00-\u9fa5，…：。！？；]")
        def extract(docs: List[str]):
            for i, doc in enumerate(docs):
                docs[i] = nc_p.sub('', doc)
        self._inner_call(extract)

    def _tokenize(self, word_level: bool = False):
        print(f"\nTokenizing in {'word' if word_level else 'char'} level for: ", end='')
        if word_level:
            def tokenize_word(docs: List[str]):
                for i, doc in enumerate(docs):
                    docs[i] = list(jieba.cut(doc))
            tokenize_fn = tokenize_word
            
        else:
            def tokenize_char(docs: List[str]):
                for i, doc in enumerate(docs):
                    docs[i] = list(doc)
            tokenize_fn = tokenize_char
        self._inner_call(tokenize_fn)

    def _add_special_tokens(self):
        print(f"\nAdding special tokens ({BLANK}, {SEPERATION}) for: ", end='')
        blank_punctuations = '，…：'
        separation_punctuations = '。！？；'
        def add_st(docs: List[List[str]]):
            for i, doc in enumerate(docs):
                n_doc = []
                for token in doc:
                    if token in blank_punctuations:
                        if n_doc[-1] != BLANK and n_doc[-1] != SEPERATION:
                            n_doc.append(BLANK)
                    elif token in separation_punctuations:
                        if n_doc[-1] != BLANK and n_doc[-1] != SEPERATION:
                            n_doc.append(SEPERATION)
                    else:
                        n_doc.append(token)
                if n_doc[-1] != SEPERATION:
                    n_doc.append(SEPERATION)
                docs[i] = n_doc
        self._inner_call(add_st)
        
    def process(self, tokenize_word_level: bool = False) -> Tuple[Tuple[List[str], List[str]], List[str]]:
        print("\nProcessing documents...")
        self._remove_ad()
        self._extract_chinese()
        self._tokenize(tokenize_word_level)
        self._add_special_tokens()

        accumulate = lambda x, y: x + y

        train_corpus = reduce(accumulate, self.train_docs)
        train_corpus_primary = train_corpus[:int(len(train_corpus) * 0.9)]
        train_corpus_heldout = train_corpus[int(len(train_corpus) * 0.9):]

        return (train_corpus_primary, 
                train_corpus_heldout) , \
                reduce(accumulate, self.test_docs)


if __name__ == "__main__":
    train_files = [r'..\resources\jyxstxtqj_downcc.com\白马啸西风.txt',
                   r'..\resources\jyxstxtqj_downcc.com\飞狐外传.txt']
    test_files = [r'..\resources\jyxstxtqj_downcc.com\碧血剑.txt']
    preprocess = PreProcess(train_files, test_files)
    train_corpus, test_corpus = preprocess.process()
    pdb.set_trace()
    