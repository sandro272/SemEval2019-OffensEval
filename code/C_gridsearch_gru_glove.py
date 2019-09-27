#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/30 9:55

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, RepeatVector, Permute, \
    TimeDistributed
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold #分层k折交叉验证
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils
from keras.models import load_model
from metrics import f1


batch_size = 128
nb_epoch = 21
hidden_dim = 120

kernel_size = 3
nb_filter = 60

train = pd.read_csv("./task_c_train/train.tsv", header=0, sep="\t", quoting=3, )
# test = pd.read_csv("./test_data/testset-taskb.tsv", header=0, sep="\t", quoting=3, )

train["subtask_c"] = train["subtask_c"].replace({"IND": 0, "GRP": 1, "OTH": 2})



def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)

    return x


def make_idx_data(revs, word_idx_map, maxlen = 150):
    """
    Transforms sentences into a 2-d matrix.
    :读取rev中的text字段，传入get_idx_from_sent()方法，将句子转换成一个list，list里面的元素是这句话每个词的索引.
    这个list形如(filter padding) - (word indices) - (Max padding) - (filter padding)，长度为max_l+2×(filter_h-1)，
    每句句子虽然本身长度不同，经过这步都转换成相同长度的list。然后，按照cv索引，分割训练集和测试集

    """
    X_train, X_test, X_dev, y_train, y_dev = [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']

        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 0:
            X_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == -1:
            X_test.append(sent)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    y_train = np_utils.to_categorical(np.array(y_train))
    y_dev = np_utils.to_categorical(np.array(y_dev))
    # y_valid = np.array(y_valid)

    return [X_train, X_test, X_dev, y_train, y_dev]

def gru_model(DROPOUT = 0.25):
    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25)(embedded)

    # bi-directional LSTM
    # hidden = Bidirectional(LSTM(hidden_dim // 2, recurrent_dropout=0.25))(embedded)

    # bi-directional GRU
    hidden = Bidirectional(GRU(hidden_dim, recurrent_dropout=DROPOUT, return_sequences=True))(embedded)
    hidden = Bidirectional(GRU(hidden_dim, recurrent_dropout=DROPOUT))(hidden)

    output = Dense(3, activation='softmax')(hidden)  # softmax, sigmoid
    rmsprop = optimizers.rmsprop(lr = 0.001)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc', f1])

    model.summary()
    return model

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    # pickle_file = sys.argv[1]
    # pickle_file = os.path.join('pickle', 'SemEval_train_val_test_TASK_C1.pickle3')
    pickle_file = os.path.join('pickle', 'SemEval_train_val_test_TASK_glove.pickle3')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, X_dev, y_train, y_dev = make_idx_data(revs, word_idx_map, maxlen = maxlen)

    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]  # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]  # 400
    logging.info("dimension of word vector [num_features]: %d" % num_features)

    model = KerasClassifier(build_fn=gru_model, nb_epoch=21, batch_size = 128, verbose=1)
    # define the grid search parameters

    # batch_size = [64, 100, 128, 256]
    # param_grid = dict(batch_size=batch_size)

    DROPOUT = [ 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    param_grid = dict(DROPOUT = DROPOUT)

    # hidden_dim = [60, 80, 100, 120, 140, 160, 180]
    # param_grid = dict(hidden_dim=hidden_dim)

    # learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    # param_grid = dict(learning_rate=learning_rate)

    kflod = KFold(n_splits=5)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kflod)
    grid_result = grid.fit(X_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))