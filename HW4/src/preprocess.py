import re
import os
import jieba


DATA_DIR = '../resources'
AUTHORS = ['jinyong', 'gulong']

def clean_and_collect(author: str):
    ad_p = re.compile(r"本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com")
    b_p = re.compile(r"\s")
    nc_p = re.compile(r"[^\u4e00-\u9fa5，…：、。！？；]")
    
    books = os.listdir(os.path.join(DATA_DIR, author))
    corpus = ''

    for book in books:
        with open(os.path.join(DATA_DIR, author, book), 'r', encoding='gb2312', errors='ignore') as fi:
            corpus += nc_p.sub('', b_p.sub('', ad_p.sub('', fi.read())))
    
    with open(os.path.join(DATA_DIR, author, 'corpus'), 'w', encoding='utf8') as fo:
        fo.write(corpus)


def tokenize(tokenize_type: str = 'char'):
    tokenized_corpus = {}
    tokens = set()
    for author in AUTHORS:
        with open(os.path.join(DATA_DIR, author, 'corpus'), 'r', encoding='utf8') as fi:
            if tokenize_type == 'char':
                tokenize_fn = list
            else:
                tokenize_fn = lambda x: list(jieba.cut(x))
            tokenized_corpus[author] = tokenize_fn(fi.read())
            tokens.update(tokenized_corpus[author])
    token2id = {token: i for i, token in enumerate(tokens)}
    id2token = {i: token for token, i in token2id.items()}
    return tokenized_corpus, token2id, id2token

if __name__ == '__main__':
    for author in AUTHORS:
        clean_and_collect(author)