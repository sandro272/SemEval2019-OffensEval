#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/28 21:25

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, SpatialDropout1D, CuDNNGRU, Flatten,Input, RepeatVector, Permute, \
    TimeDistributed
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold, KFold #分层k折交叉验证
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils
from keras.models import load_model
from metrics import f1
from Attention_layer import AttentionM
from capsule_net import Capsule


train = pd.read_csv("./task_c_train/train.tsv", header=0, sep="\t", quoting=3, )
test = pd.read_csv("./test_data/test_set_taskc.tsv", header=0, sep="\t", quoting=3, )


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


def make_idx_data(revs, word_idx_map, maxlen=150):
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

def bi_lstm_glove(batch_size = 64, nb_epoch = 27, hidden_dim = 140):
    sequence = Input(shape=(maxlen1,), dtype='int32')

    embedded = Embedding(input_dim = W1.shape[0], output_dim=W1.shape[1], input_length = maxlen1, mask_zero=True,
                         weights=[W1], trainable=False)(sequence)

    embedded = Dropout(0.25)(embedded)

    embedded = Bidirectional(LSTM(hidden_dim, dropout=0.35, return_sequences=True))(embedded)
    enc = Bidirectional(LSTM(hidden_dim, dropout=0.35))(embedded)

    fc1 = Dense(128, activation="relu")(enc)
    fc2_dropout = Dropout(0.3)(fc1)

    output = Dense(3, activation='sigmoid')(fc2_dropout)
    rmsprop = optimizers.rmsprop(lr=0.001)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc', f1])
    class_weight = {0: 1, 1: 2, 2: 6}

    train_num, test_num = X_train1.shape[0], X_test1.shape[0]
    # train_num, test_num = X_train1.shape[0], X_dev1.shape[0]
    num1 = y_train1.shape[1]

    second_level_train_set = np.zeros((train_num, num1))

    second_level_test_set = np.zeros((test_num, num1))

    test_nfolds_sets = []

    # kf = KFold(n_splits = 2)
    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train1)):
        x_tra, y_tra = X_train1[train_index], y_train1[train_index]

        x_tst, y_tst = X_train1[test_index], y_train1[test_index]


        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight)

        second_level_train_set[test_index] = model.predict(x_tst, batch_size=batch_size)

        test_nfolds_sets.append(model.predict(X_test1))
        # test_nfolds_sets.append(model.predict(X_dev1))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    return second_level_train_set ,second_level_test_set


def bi_lstm_elmo(batch_size = 128, nb_epoch = 28, hidden_dim = 40):

    sequence = Input(shape=(maxlen2,), dtype='int32')

    embedded = Embedding(input_dim=W2.shape[0], output_dim=W2.shape[1], input_length=maxlen2, mask_zero=True,
                         weights=[W2], trainable=False)(sequence)

    embedded = Dropout(0.25)(embedded)


    embedded = Bidirectional(LSTM(hidden_dim, dropout=0.25, return_sequences=True))(embedded)
    enc = Bidirectional(LSTM(hidden_dim, dropout=0.25))(embedded)

    fc1 = Dense(128, activation="relu")(enc)
    fc2_dropout = Dropout(0.3)(fc1)

    output = Dense(3, activation='sigmoid')(fc2_dropout)
    rmsprop = optimizers.rmsprop(lr=0.01)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc', f1])

    class_weight = {0: 1, 1: 2, 2: 6}

    train_num, test_num = X_train2.shape[0], X_test2.shape[0]
    # train_num, test_num = X_train2.shape[0], X_dev2.shape[0]
    num1 = y_train2.shape[1]

    second_level_train_set = np.zeros((train_num, num1))

    second_level_test_set = np.zeros((test_num, num1))

    test_nfolds_sets = []

    # kf = KFold(n_splits = 2)
    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train2)):
        x_tra, y_tra = X_train2[train_index], y_train2[train_index]

        x_tst, y_tst = X_train2[test_index], y_train2[test_index]

        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight)

        second_level_train_set[test_index] = model.predict(x_tst, batch_size=batch_size)

        test_nfolds_sets.append(model.predict(X_test2))
        # test_nfolds_sets.append(model.predict(X_dev2))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    return second_level_train_set, second_level_test_set

def gru_glove(batch_size = 128, nb_epoch = 21, hidden_dim = 120):
    sequence = Input(shape=(maxlen1,), dtype='int32')

    embedded = Embedding(input_dim=W1.shape[0], output_dim=W1.shape[1], input_length=maxlen1, mask_zero=True,
                         weights=[W1], trainable=False)(sequence)
    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25)(embedded)


    hidden = Bidirectional(GRU(hidden_dim, recurrent_dropout=0.25, return_sequences=True))(embedded)
    hidden = Bidirectional(GRU(hidden_dim, recurrent_dropout=0.25))(hidden)

    output = Dense(3, activation='softmax')(hidden)  # softmax, sigmoid
    rmsprop = optimizers.rmsprop(lr=0.001)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc', f1])

    class_weight = {0: 1, 1: 2, 2: 6}

    train_num, test_num = X_train1.shape[0], X_test1.shape[0]
    # train_num, test_num = X_train1.shape[0], X_dev1.shape[0]
    num1 = y_train1.shape[1]

    second_level_train_set = np.zeros((train_num, num1))

    second_level_test_set = np.zeros((test_num, num1))

    test_nfolds_sets = []

    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train1)):
        x_tra, y_tra = X_train1[train_index], y_train1[train_index]

        x_tst, y_tst = X_train1[test_index], y_train1[test_index]

        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight)

        second_level_train_set[test_index] = model.predict(x_tst, batch_size=batch_size)

        test_nfolds_sets.append(model.predict(X_test1))
        # test_nfolds_sets.append(model.predict(X_dev1))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    return second_level_train_set, second_level_test_set

def gru_elmo(batch_size = 64, nb_epoch = 22, hidden_dim = 60):
    sequence = Input(shape=(maxlen2,), dtype='int32')

    embedded = Embedding(input_dim=W2.shape[0], output_dim=W2.shape[1], input_length=maxlen2, mask_zero=True,
                         weights=[W2], trainable=False)(sequence)

    embedded = Dropout(0.25)(embedded)


    hidden = Bidirectional(GRU(hidden_dim, recurrent_dropout=0.25, return_sequences=True))(embedded)
    hidden = Bidirectional(GRU(hidden_dim, recurrent_dropout=0.25))(hidden)

    output = Dense(3, activation='softmax')(hidden)  # softmax, sigmoid
    rmsprop = optimizers.rmsprop(lr=0.001)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc', f1])

    class_weight = {0: 1, 1: 2, 2: 6}

    train_num, test_num = X_train2.shape[0], X_test2.shape[0]
    # train_num, test_num = X_train2.shape[0], X_dev2.shape[0]
    num1 = y_train2.shape[1]

    second_level_train_set = np.zeros((train_num, num1))

    second_level_test_set = np.zeros((test_num, num1))

    test_nfolds_sets = []

    # kf = KFold(n_splits = 2)
    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train2)):
        x_tra, y_tra = X_train2[train_index], y_train2[train_index]

        x_tst, y_tst = X_train2[test_index], y_train2[test_index]

        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight)

        second_level_train_set[test_index] = model.predict(x_tst, batch_size=batch_size)

        test_nfolds_sets.append(model.predict(X_test2))
        # test_nfolds_sets.append(model.predict(X_dev2))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    return second_level_train_set,second_level_test_set

def attention_glove(batch_size = 128, nb_epoch = 24, hidden_dim = 180):
    sequence = Input(shape=(maxlen1,), dtype='int32')

    embedded = Embedding(input_dim=W1.shape[0], output_dim=W1.shape[1], input_length=maxlen1, mask_zero=True,
                         weights=[W1], trainable=False)(sequence)

    embedded = Dropout(0.25)(embedded)


    enc = Bidirectional(GRU(hidden_dim, dropout=0.3, return_sequences=True))(embedded)
    enc = Bidirectional(GRU(hidden_dim, dropout=0.3, return_sequences=True))(enc)

    att = AttentionM()(enc)

    fc1_dropout = Dropout(0.25)(att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.25)(fc1)

    output = Dense(3, activation='softmax')(fc2_dropout)

    model = Model(inputs=sequence, outputs=output)

    rmsprop = optimizers.rmsprop(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc', f1])

    class_weight = {0: 1, 1: 2, 2: 6}

    train_num, test_num = X_train1.shape[0], X_test1.shape[0]
    # train_num, test_num = X_train1.shape[0], X_dev1.shape[0]
    num1 = y_train1.shape[1]

    second_level_train_set = np.zeros((train_num, num1))

    second_level_test_set = np.zeros((test_num, num1))

    test_nfolds_sets = []

    # kf = KFold(n_splits = 2)
    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train1)):
        x_tra, y_tra = X_train1[train_index], y_train1[train_index]

        x_tst, y_tst = X_train1[test_index], y_train1[test_index]

        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight)

        second_level_train_set[test_index] = model.predict(x_tst, batch_size=batch_size)

        test_nfolds_sets.append(model.predict(X_test1))
        # test_nfolds_sets.append(model.predict(X_dev1))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    return second_level_train_set, second_level_test_set


def attention_elmo(batch_size = 128, nb_epoch = 35, hidden_dim = 80):
    sequence = Input(shape=(maxlen2,), dtype='int32')

    embedded = Embedding(input_dim=W2.shape[0], output_dim=W2.shape[1], input_length=maxlen2, mask_zero=True,
                         weights=[W2], trainable=False)(sequence)

    embedded = Dropout(0.25)(embedded)

    enc = Bidirectional(GRU(hidden_dim, dropout=0.35, return_sequences=True))(embedded)
    enc = Bidirectional(GRU(hidden_dim, dropout=0.35, return_sequences=True))(enc)

    att = AttentionM()(enc)


    fc1_dropout = Dropout(0.25)(att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.25)(fc1)

    output = Dense(3, activation='softmax')(fc2_dropout)

    model = Model(inputs=sequence, outputs=output)

    rmsprop = optimizers.rmsprop(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc', f1])

    class_weight = {0: 1, 1: 2, 2: 6}

    train_num, test_num = X_train2.shape[0], X_test2.shape[0]
    # train_num, test_num = X_train2.shape[0], X_dev2.shape[0]
    num1 = y_train2.shape[1]

    second_level_train_set = np.zeros((train_num, num1))

    second_level_test_set = np.zeros((test_num, num1))

    test_nfolds_sets = []

    # kf = KFold(n_splits = 2)
    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train2)):
        x_tra, y_tra = X_train2[train_index], y_train2[train_index]

        x_tst, y_tst = X_train2[test_index], y_train2[test_index]

        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight)

        second_level_train_set[test_index] = model.predict(x_tst, batch_size=batch_size)

        test_nfolds_sets.append(model.predict(X_test2))
        # test_nfolds_sets.append(model.predict(X_dev2))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    return second_level_train_set, second_level_test_set

def capsulenet_glove(batch_size = 64, nb_epoch = 29, hidden_dim = 140):
    Routings = 15
    Num_capsule = 30
    Dim_capsule = 60

    sequence_input = Input(shape=(maxlen1,), dtype='int32')
    embedded_sequences = Embedding(input_dim=W1.shape[0], output_dim=W1.shape[1], input_length=maxlen1, weights=[W1],
                                   trainable=False)(sequence_input)
    embedded_sequences = SpatialDropout1D(0.1)(embedded_sequences)
    x = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))(embedded_sequences)
    x = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True,
                      kernel_size=(3, 1))(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(0.4)(capsule)
    output = Dense(3, activation='softmax')(capsule)
    model = Model(inputs=[sequence_input], outputs=output)

    rmsprop = optimizers.rmsprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy', f1])

    class_weight = {0: 1, 1: 2, 2: 6}

    train_num, test_num = X_train1.shape[0], X_test1.shape[0]
    # train_num, test_num = X_train1.shape[0], X_dev1.shape[0]
    num1 = y_train1.shape[1]

    second_level_train_set = np.zeros((train_num, num1))

    second_level_test_set = np.zeros((test_num, num1))

    test_nfolds_sets = []

    # kf = KFold(n_splits = 2)
    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train1)):
        x_tra, y_tra = X_train1[train_index], y_train1[train_index]

        x_tst, y_tst = X_train1[test_index], y_train1[test_index]

        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight)

        second_level_train_set[test_index] = model.predict(x_tst, batch_size=batch_size)

        test_nfolds_sets.append(model.predict(X_test1))
        # test_nfolds_sets.append(model.predict(X_dev1))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    return second_level_train_set, second_level_test_set


def capsulnet_elmo(batch_size = 64, nb_epoch = 40, hidden_dim = 80):
    Routings = 15
    Num_capsule = 30
    Dim_capsule = 60

    sequence_input = Input(shape=(maxlen2,), dtype='int32')
    embedded_sequences = Embedding(input_dim=W2.shape[0], output_dim=W2.shape[1], input_length=maxlen2, weights=[W2],
                                   trainable=False)(sequence_input)
    embedded_sequences = SpatialDropout1D(0.1)(embedded_sequences)
    x = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))(embedded_sequences)
    x = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True,
                      kernel_size=(3, 1))(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(0.4)(capsule)
    output = Dense(3, activation='softmax')(capsule)
    model = Model(inputs=[sequence_input], outputs=output)

    rmsprop = optimizers.rmsprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy', f1])

    class_weight = {0: 1, 1: 2, 2: 6}

    train_num, test_num = X_train2.shape[0], X_test2.shape[0]
    # train_num, test_num = X_train2.shape[0], X_dev2.shape[0]
    num1 = y_train2.shape[1]

    second_level_train_set = np.zeros((train_num, num1))

    second_level_test_set = np.zeros((test_num, num1))

    test_nfolds_sets = []

    # kf = KFold(n_splits = 2)
    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train2)):
        x_tra, y_tra = X_train2[train_index], y_train2[train_index]

        x_tst, y_tst = X_train2[test_index], y_train2[test_index]

        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight)

        second_level_train_set[test_index] = model.predict(x_tst, batch_size=batch_size)

        test_nfolds_sets.append(model.predict(X_test2))
        # test_nfolds_sets.append(model.predict(X_dev2))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    return second_level_train_set, second_level_test_set

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')

    pickle_file1 = os.path.join('pickle', 'SemEval_train_val_test_TASK_C_glove_terminal.pickle3')
    # # pickle_file_glove = load_data_glove()
    #
    revs1, W1, word_idx_map1, vocab1, maxlen1 = pickle.load(open(pickle_file1, 'rb'))
    logging.info('data loaded!')
    #
    X_train1, X_test1, X_dev1, y_train1, y_dev1 = make_idx_data(revs1, word_idx_map1, maxlen=maxlen1)

    pickle_file2 = os.path.join('pickle', 'SemEval_train_val_test_TASK_C_elmo_terminal.pickle3')
    #
    revs2, W2, word_idx_map2, vocab2, maxlen2 = pickle.load(open(pickle_file2, 'rb'))
    logging.info('data loaded!')
    #
    X_train2, X_test2, X_dev2, y_train2, y_dev2 = make_idx_data(revs2, word_idx_map2, maxlen=maxlen2)

    train_sets = []
    test_sets = []
    # dev_sets = []
    for clf in [
                bi_lstm_glove(batch_size = 64, nb_epoch = 27, hidden_dim = 140),
                gru_glove(batch_size=128, nb_epoch=21, hidden_dim=120),
                attention_glove(batch_size=128, nb_epoch=24, hidden_dim=180),
                capsulenet_glove(batch_size=64, nb_epoch=29, hidden_dim=140),
                ]:
        train_set, test_set = clf
        # train_set, dev_set = clf
        train_sets.append(train_set)
        test_sets.append(test_set)
        # dev_sets.append(dev_set)

    meta_train = np.concatenate([result_set.reshape(-1, 3) for result_set in train_sets], axis=1)
    # print("meta_train:", meta_train)
    meta_test = np.concatenate([y_test_set.reshape(-1, 3) for y_test_set in test_sets], axis=1)
    # meta_dev = np.concatenate([y_dev_set.reshape(-1, 3) for y_dev_set in dev_sets], axis=1)
    # print("meta_dev:",meta_dev)

    # 使用决策树作为我们的次级分类器
    from sklearn.tree import DecisionTreeClassifier

    dt_model = DecisionTreeClassifier()
    dt_model.fit(meta_train, y_train1)
    df_predict = dt_model.predict(meta_test)
    # df_predict = dt_model.predict(meta_dev)
    # print("df_predict:", df_predict)
    y_pred = np.argmax(df_predict, axis=1)
    print("y_pred:",y_pred)
    # y_dev = np.argmax(y_dev1, axis=1)
    # print("y_dev:",y_dev)
    result_output = pd.DataFrame(data={"id": test["id"], "label": y_pred}, )
    # #
    result_output.to_csv("./result/task_C1_submission_stacking.csv", index=False, quoting=3, )

    # from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    # print(precision_score(y_dev, df_predict, average='macro'))
    # print(recall_score(y_dev, df_predict, average='macro'))
    # print(accuracy_score(y_dev, df_predict))
    # print(f1_score(y_dev, df_predict, average='macro'))
    # print(precision_score(y_dev, y_pred, average='macro'))
    # print(recall_score(y_dev, y_pred, average='macro'))
    # print(accuracy_score(y_dev, y_pred))
    # print(f1_score(y_dev, y_pred, average='macro'))





