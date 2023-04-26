import random
import jieba
import pdb
import re
import os
from typing import Tuple, List, Callable, Union

random.seed(0)

class PreProcess:

    def __init__(self, data_dir: str, catalog_name: str, stopword_name: str, tokenize_mode='char'):
        self.data_dir = data_dir
        with open(os.path.join(self.data_dir, catalog_name), 'r', encoding='gb2312', errors='ignore') as fi:
            self.book2index = {book: i for i, book in enumerate(fi.read().strip().split(','))}
            self.index2book = {i: book for book, i in self.book2index.items()}
            print("Available books and their indices:")
            for book, index in self.book2index.items():
                print(f"{book}: {index}")
        self.stopword_path = os.path.join(data_dir, stopword_name)
        self.tokenize_mode = tokenize_mode
        self._read_books()
        self._tokenize()

    def _read_books(self):
        self.book2content = {}
        ad_p = re.compile(r"本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com")
        b_p = re.compile(r"\s")
        nc_p = re.compile(r"[^\u4e00-\u9fa5，…：、。！？；]")
        print('Read and parse books:')
        for book in self.book2index:
            print(f'{book}, ', end='')
            with open(os.path.join(self.data_dir, f'{book}.txt'), 'r', encoding='gb2312', errors='ignore') as fi:
                content = fi.read()
                content = ad_p.sub('', content)
                lines = [nc_p.sub('', b_p.sub('', line)) for line in content.strip().split('\n')]
                self.book2content[book] = lines
        print('Done reading and parsing.')


    def _tokenize(self):
        print(f"Tokenize in {self.tokenize_mode} mode:") 
        tokens = set()
        self.book2token_count = {}
        for book, content in self.book2content.items():
            tokenized_content = []
            token_count = 0
            for para in content:
                if self.tokenize_mode == 'char':
                    tokenized_content.append(list(para))
                else:
                    tokenized_content.append(list(jieba.cut(para)))
                token_count += len(tokenized_content[-1])
                tokens.update(tokenized_content[-1])
            self.book2content[book] = tokenized_content
            self.book2token_count[book] = token_count
            print(f"Token number of {book}: {token_count}")
        self.token2idx = {token: i for i, token in enumerate(tokens)}
        self.idx2token = {i: token for token, i in self.token2idx.items()}
        print(f"Vocabulary size: {len(tokens)}")
    
    def sample_corpus(self, para_num: int = 200, min_token_num: int = 500, max_token_num: int = 1000):
        book_seq = []
        weights = []
        for book, token_count in self.book2token_count.items():
            book_seq.append(book)
            weights.append(token_count)
        book_para4sample = {}
        for book in self.book2token_count:
            book_para4sample[book] = []
            for para_idx, para in enumerate(self.book2content[book]):
                book_para4sample[book].append(len(para))
        corpus = []
        while len(corpus) < para_num:
            book = random.choices(book_seq, weights=weights)[0]
            para_idx = random.choices(range(len(book_para4sample[book])), weights=book_para4sample[book])[0]
            para_tokens = self.book2content[book][para_idx]
            while len(para_tokens) < min_token_num and para_idx < len(book_para4sample[book]) - 1:
                para_idx += 1
                para_tokens.extend(self.book2content[book][para_idx])
            if len(para_tokens) >= min_token_num and len(para_tokens) <= max_token_num:
                corpus.append((para_tokens, book))
        idx_corpus = []
        for para, book in corpus:
            idx_corpus.append(([self.token2idx[token] for token in para], self.book2index[book]))
        return idx_corpus
