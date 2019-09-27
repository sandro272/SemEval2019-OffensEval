#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/18 15:38

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging
import re
import nltk
import gensim
import pickle

import numpy as np
import pandas as pd

from preprocess_demo import *
from ekphrasis_tool import ekphrasis_config
from stanford_tokenizer_config import *
from collections import defaultdict

train = pd.read_csv("./task_b_train/train1.tsv", header=0, sep="\t", quoting=3, )
test = pd.read_csv("./test_data/testset-taskb.tsv", header=0, sep="\t", quoting=3, )

# print(train.columns)
# print(train.columns)
# print(train.subtask_a)
# print(train.dtypes)

train["subtask_b"] = train["subtask_b"].replace({"TIN": 0, "UNT": 1})



def review_to_wordlist(review_text):

    review_text = emoji_to_text(review_text)

    review_text = abbreviation_to_text(review_text)

    review_text = re.sub("(@[\w]*\ )+", "@USER", review_text)

    duplicateSpacePattern = re.compile(r'\ +')
    review_text = re.sub(duplicateSpacePattern, ' ', review_text).strip()

    # review_text = re.sub("@[\w]*\ ", " ", review_text)

    # review_text = re.sub("(@[\w]*\ )+", "@USER ", review_text).strip()     #将重复出现的@USER替换成只有一个的@USER
    # print(review_text)

    # review_text = re.sub("[!?,.]", " ", review_text).strip()

    review_text = ekphrasis_config(review_text)

    review_text = re.sub("[^a-zA-Z0-9\@]", " ", str(review_text))

    # review_text = review_text.lower()
    # print(review_text)
    words = stanford_tokenizer(review_text)

    # return (review_text)
    return (words)


def build_data_train_test(data_train, data_test, train_ratio=0.8):
    """
    Loads data and process data into index
    """
    revs = []
    vocab = defaultdict(float)    #  一个dict，key是数据集中出现的word，value是该word出现的次数

    # Pre-process train data set
    for i in range(len(data_train)):

        print("第%d条,共%d条" % (i, len(data_train)))

        rev = data_train[i]
        y = train["subtask_b"][i]
        # print("rev:",rev)
        # print("y:",y)
        # rev = rev.strip()
        orig_rev = ' '.join(rev).lower()
        # orig_rev = review_to_wordlist(rev)

        # print(type(orig_rev))                    #list
        # orig_rev = str(orig_rev)

        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': y,        # y是类标
                 'text': orig_rev,      # text是句子原文（经过清洗）
                 'num_words': len(orig_rev.split()),      # 句子长度（词数）
                 'split': int(np.random.rand() < train_ratio)}  # 分配的索引
        revs.append(datum)
        # print("datum:",datum)

    for i in range(len(data_test)):

        # print("第%d条,共%d条" % (i, len(data_test)))

        rev = data_test[i]
        orig_rev = ' '.join(rev).lower()
        # orig_rev = review_to_wordlist(rev)
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': -1,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': -1}
        revs.append(datum)

    return revs, vocab


def load_bin_vec(model, vocab):
    '''
    从GoogleNews-vectors-negative300.bin中加载w2v矩阵。生成w2v。w2v是一个dict，key是word，value是vector
    '''
    word_vecs = {}
    unk_words = 0

    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unk_words = unk_words + 1

    logging.info('unk words: %d' % (unk_words))
    return word_vecs


def get_W(word_vecs, k=300):
    '''
    接收w2v，相当于把w2v从字典转换成矩阵W，并且生成word_idx_map。
    相当于原来从word到vector只用查阅w2v字典；现在需要先从word_idx_map查阅word的索引，再2用word的索引到W矩阵获取vector
    '''
    vocab_size = len(word_vecs)
    word_idx_map = dict()                        #  一个dict，key是数据集中出现的word，value是该word的索引

    W = np.zeros(shape=(vocab_size + 2, k), dtype=np.float32)         #  即word matrix，W[i]是索引为i的词对应的词向量
    W[0] = np.zeros((k,))
    W[1] = np.random.uniform(-0.25, 0.25, k)

    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map     #         W2V类似于W，但是是随机初始化的


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    clean_train_reviews = []
    for review in train["tweet"]:
        clean_train_reviews.append(review_to_wordlist(review))

    clean_test_reviews = []
    # for review in test["tweet"]:
    #     clean_test_reviews.append(review_to_wordlist(review))

    revs, vocab = build_data_train_test(clean_train_reviews, clean_test_reviews)
    max_l = np.max(pd.DataFrame(revs)['num_words'])
    logging.info('data loaded!')
    logging.info('number of sentences: ' + str(len(revs)))
    logging.info('vocab size: ' + str(len(vocab)))
    logging.info('max sentence length: ' + str(max_l))

    # word2vec GoogleNews
    # model_file = os.path.join('vector', 'GoogleNews-vectors-negative300.bin')
    # model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    # Glove Common Crawl
    model_file = os.path.join('vector', 'glove_model.txt')
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)

    w2v = load_bin_vec(model, vocab)
    logging.info('word embeddings loaded!')
    logging.info('num words in embeddings: ' + str(len(w2v)))

    W, word_idx_map = get_W(w2v, k=model.vector_size)
    logging.info('extracted index from embeddings! ')

    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    pickle_file = os.path.join('pickle', 'SemEval_train_val_test_TASK_B_TRAIL1.pickle3')
    # pickle_file = os.path.join('pickle', 'SemEval_train_val_test.pickle3')
    pickle.dump([revs, W, word_idx_map, vocab, max_l], open(pickle_file, 'wb'))
    logging.info('dataset created!')
