#! /usr/bin/env python

import os
import sys
import logging
import re
import nltk
import gensim
import pickle

import numpy as np
import pandas as pd

from stanford_tokenizer_config import *
from collections import defaultdict

train = pd.read_csv("./train_data/train_subtask_a", header=0, sep="\t", quoting=3, )
# test = pd.read_csv("./test_data/test_subtask_a", header = 0, sep = "\t", quoting = 3, )

# print(train.columns)
# print(train.columns)
# print(train.subtask_a)
# print(train.dtypes)

train["subtask_a"] = train["subtask_a"].replace({"NOT": 0, "OFF": 1})


# train["subtask_b"] = train["subtask_b"].replace({"TIN":0, "UNT":1})
# train["subtask_c"] = train["subtask_c"].replace({"IND":0, "GRP":1, "OTH":2})

# print(train["subtask_a"])

def remove_pattern(input_text, pattern):
    res = re.findall(pattern, input_text)
    for i in res:
        input_text = re.sub(i, '', input_text)
    return input_text



def review_to_wordlist(review_text):

    # review_text = review_text.replace({"👊", "Oncoming Fist"})


    review_text = remove_pattern(str(review_text), "@[\w]*").strip()

    review_text = review_text.lower()

    words = stanford_tokenizer(review_text)

    # return (review_text)
    return (words)


def build_data_train_test(data_train, data_test, train_ratio=0.8):
    """
    Loads data and process data into index
    """
    revs = []
    vocab = defaultdict(float)

    # Pre-process train data set
    for i in range(len(data_train)):

        # print("第%d条,共%d条" % (i,len(data_train)))

        rev = data_train[i]
        y = train["subtask_a"][i]

        # orig_rev = ' '.join(rev).lower()
        orig_rev = review_to_wordlist(rev)

        print(type(orig_rev))

        words = set(orig_rev.split)
        for word in words:
            vocab[word] += 1
        datum = {'y': y,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': int(np.random.rand() < train_ratio)}
        revs.append(datum)

    for i in range(len(data_test)):

        # print("第%d条,共%d条" % (i, len(data_test)))

        rev = data_test[i]
        # orig_rev = ' '.join(rev).lower()
        orig_rev = review_to_wordlist(rev)
        words = set(orig_rev)
        for word in words:
            vocab[word] += 1
        datum = {'y': -1,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': -1}
        revs.append(datum)

    return revs, vocab


def load_bin_vec(model, vocab):
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
    vocab_size = len(word_vecs)
    word_idx_map = dict()

    W = np.zeros(shape=(vocab_size + 2, k), dtype=np.float32)
    W[0] = np.zeros((k,))
    W[1] = np.random.uniform(-0.25, 0.25, k)

    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map


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
    # for Content in test["tweet"]:
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
    pickle_file = os.path.join('pickle', 'SemEval_train_val_test.pickle3')
    pickle.dump([revs, W, word_idx_map, vocab, max_l], open(pickle_file, 'wb'))
    logging.info('dataset created!')

