import random
import pdb
import os
import sys
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Callable, Union
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from lda import LDA
from preprocess import PreProcess

random.seed(0)

class Classification:
    def __init__(self, features: np.ndarray, target: List[int], test_size: float = 0.3):
        self.clf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')

        print('Splitting data...')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target, test_size=test_size, random_state=0)

        print('Training...')
        self.clf.fit(self.X_train, self.y_train)

        print('Predicting...')
        self.y_pred = self.clf.predict(self.X_test)
    
    def get_accuracy(self):
        return accuracy_score(self.y_test, self.y_pred)
    
    def get_f1(self):
        return f1_score(self.y_test, self.y_pred, average='macro')
    
    def get_precision(self):
        return precision_score(self.y_test, self.y_pred, average='macro')
    
    def get_recall(self):
        return recall_score(self.y_test, self.y_pred, average='macro')

if __name__ == '__main__':
    '''
    models = {'char': {}, 'word': {}}
    metrics = {'char': {}, 'word': {}}
    best_model = None
    best_mode = None
    best_K = None
    best_metric = float('-inf')
    print("=============== CHAR ===============")
    preprocess = PreProcess('resources', 'inf.txt', 'cn_stopwords.txt', tokenize_mode='char')
    corpus = preprocess.sample_corpus(para_num=200, min_token_num=500)
    V = len(preprocess.token2idx)
    print("--------------- RESULT ---------------")
    for K in range(40, 201, 40):
        lda = LDA(alpha=np.ones((K,)), beta=np.ones((V,)))
        lda.fit(corpus, max_iter=100, min_topic_change=100)
        
        models['char'][K] = lda
        
        theta = lda.get_theta()
        clf = Classification(theta, [label for _, label in corpus], test_size=0.3)
        acc = clf.get_accuracy()
        
        metrics['char'][K] = acc
        print(f"K = {K}: {metrics['char'][K]}")
        if acc > best_metric:
            best_model = lda
            best_mode = 'char'
            best_K = K
        

    print("=============== WORD ===============")
    preprocess = PreProcess('resources', 'inf.txt', 'cn_stopwords.txt', tokenize_mode='word')
    corpus = preprocess.sample_corpus(para_num=200, min_token_num=500)
    V = len(preprocess.token2idx)
    print("--------------- RESULT ---------------")
    for K in range(40, 201, 40):
        lda = LDA(alpha=np.ones((K,)), beta=np.ones((V,)))
        lda.fit(corpus, max_iter=100, min_topic_change=100)
        
        models['word'][K] = lda
        
        theta = lda.get_theta()
        clf = Classification(theta, [label for _, label in corpus], test_size=0.3)
        acc = clf.get_accuracy()
        
        metrics['word'][K] = acc

        print(f"K = {K}: {metrics['word'][K]}")

        if acc > best_metric:
            best_model = lda
            best_mode = 'word'
            best_K = K
    
    print("=============== BEST ===============")
    print(f"Best mode: {best_mode}")
    print(f"Best K: {best_K}")
    print(f"Best accuracy: {best_metric}")

    print("=============== TEST ===============")
    '''
    tokenize_mode = sys.argv[1]
    K = int(sys.argv[2])
    preprocess = PreProcess('resources', 'inf.txt', 'cn_stopwords.txt', tokenize_mode=tokenize_mode)
    corpus = preprocess.sample_corpus(para_num=200, min_token_num=500)
    V = len(preprocess.token2idx)
    lda = LDA(alpha=np.ones((K,)), beta=np.ones((V,)))
    lda.fit(corpus, max_iter=100, min_topic_change=100)
    theta = lda.get_theta()
    clf = Classification(theta, [label for _, label in corpus], test_size=0.3)
    acc = clf.get_accuracy()
    print(f"K = {K}: {acc}")
    pdb.set_trace()