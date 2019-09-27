#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/23 8:21

import os
import sys
import logging
import pandas as pd
import numpy as np
import pickle

import numpy as np
from keras.preprocessing import sequence
from sklearn.model_selection import KFold
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, Convolution1D, MaxPooling1D, Flatten, CuDNNGRU, SpatialDropout1D, TimeDistributed
from metrics import f1
from keras.callbacks import ModelCheckpoint,EarlyStopping
from Attention_layer import AttentionM
from capsule_net import Capsule


train = pd.read_csv("./task_b_train/train1.tsv", header=0, sep="\t", quoting=3, )
# test = pd.read_csv("./test_data/testset-taska.tsv", header=0, sep="\t", quoting=3, )

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


def make_idx_data(revs, word_idx_map, maxlen = 120):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, X_dev, y_train, y_dev,= [], [], [], [], []
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
    y_train = np_utils.to_categorical(np.array(y_train))
    y_dev = np_utils.to_categorical(np.array(y_dev))
    # print(X_train)
    # print(y_train)
    # print(X_test)
    # print(X_dev)
    # print(y_dev)
    return [X_train, X_test, X_dev, y_train, y_dev,]

def bi_lstm_model():
    batch_size = 256
    nb_epoch = 40
    hidden_dim = 120

    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25)(embedded)

    # bi-directional LSTM
    hidden = Bidirectional(LSTM(hidden_dim//2, recurrent_dropout=0.25)) (embedded)

    # bi-directional GRU
    # hidden = Bidirectional(GRU(hidden_dim // 2, recurrent_dropout=0.25))(embedded)

    output = Dense(2, activation='softmax')(hidden)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])

    model.summary()
    return model

def bi_gru_model():
    batch_size = 100
    nb_epoch = 40
    hidden_dim = 120

    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25)(embedded)

    # bi-directional LSTM
    # hidden = Bidirectional(LSTM(hidden_dim//2, recurrent_dropout=0.25)) (embedded)

    # bi-directional GRU
    hidden = Bidirectional(GRU(hidden_dim // 2, recurrent_dropout=0.25))(embedded)

    output = Dense(2, activation='softmax')(hidden)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])

    model.summary()
    return model

def attention_bi_lstm_model():
    batch_size = 256
    nb_epoch = 40
    hidden_dim = 120

    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25)(embedded)

    # bi-lstm
    enc = Bidirectional(LSTM(hidden_dim // 2, recurrent_dropout=0.25, return_sequences=True))(embedded)

    # gru
    # enc = Bidirectional(GRU(hidden_dim//2, recurrent_dropout=0.2, return_sequences=True)) (embedded)

    att = AttentionM()(enc)

    # print(enc.shape)
    # print(att.shape)

    fc1_dropout = Dropout(0.25)(att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.25)(fc1)

    output = Dense(2, activation='softmax')(fc2_dropout)
    model = Model(inputs=sequence, outputs=output)
    class_weight = {0: 1, 1: 7}

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])

    model.summary()
    return model

def capsulnet_model():
    batch_size = 100
    nb_epoch = 40
    hidden_dim = 120

    Routings = 15
    Num_capsule = 30
    Dim_capsule = 60

    sequence_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W],
                                   trainable=False)(sequence_input)
    embedded_sequences = SpatialDropout1D(0.1)(embedded_sequences)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(embedded_sequences)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(0.4)(capsule)
    output = Dense(2, activation='softmax')(capsule)
    model = Model(inputs=[sequence_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])

    model.summary()
    return model

def stacked_bi_lstm_model():
    batch_size = 100
    nb_epoch = 40
    hidden_dim = 120

    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True, weights=[W], trainable=False)(sequence)
    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25)(embedded)

    # bi-directional LSTM
    hidden = Bidirectional(LSTM(hidden_dim // 2, recurrent_dropout=0.25, return_sequences=True))(embedded)
    # hidden = Bidirectional(LSTM(hidden_dim // 2, recurrent_dropout=0.25, return_sequences=True))(hidden)
    hidden = Bidirectional(LSTM(hidden_dim // 2, recurrent_dropout=0.25))(hidden)

    # bi-directional GRU
    # hidden = Bidirectional(GRU(hidden_dim//2, returrent_dropout=0.25, return_sequences=True)) (embedded)
    # hidden = Bidirectional(GRU(hidden_dim//2, recurrent_dropout=0.25)) (hidden)

    output = Dense(2, activation='softmax')(hidden)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])

    model.summary()
    return model


def get_stacking(clf, x_train, y_train, x_test, n_folds = 5):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""


    train_num, test_num = x_train.shape[0], x_test.shape[0]
    num = y_train.shape[1]


    second_level_train_set = np.zeros((train_num, num))                     # (10556,)

    second_level_test_set = np.zeros((test_num,num))                       # (2684,)

    class_weight = {0: 1, 1: 7}
    test_nfolds_sets = []

    kf = KFold(n_splits=n_folds)

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):


        x_tra, y_tra = x_train[train_index], y_train[train_index]

        x_tst, y_tst = x_train[test_index], y_train[test_index]

        early_stopping = EarlyStopping(monitor='val_acc', patience=12, verbose=1)

        clf.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight, callbacks=[early_stopping])

        # clf.fit(x_tra, y_tra, class_weight = class_weight, batch_size = batch_size, epochs = nb_epoch, verbose = 2)

        second_level_train_set[test_index] = clf.predict(x_tst, batch_size = batch_size)     #  (2112,2) could not be broadcast to indexing result of shape (2112,)

        test_nfolds_sets.append(clf.predict(x_test))

    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / n_folds

    return second_level_train_set, second_level_test_set




if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    # pickle_file = sys.argv[1]
    pickle_file = os.path.join('pickle', 'SemEval_train_val_test_TASK_B_TRAIL2.pickle3')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, X_dev, y_train, y_dev = make_idx_data(revs, word_idx_map, maxlen = maxlen)

    # print("X_train", X_train.shape)  # (3546, 102)
    # print("y_train", y_train.shape)  # (3546, 2)
    # print("X_dev", X_dev.shape)  # (854, 102)
    # print(" y_dev", y_dev.shape)  # (854, 2)

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

    nb_epoch_h = 40
    # nb_epoch_h =  [1,1,1,1,1]
    batch_size_z = [256, 100, 256, 100, 100]
    j = 0
    train_sets = []
    # test_sets = []
    dev_sets = []
    for clf in [bi_lstm_model(), bi_gru_model(), attention_bi_lstm_model(), capsulnet_model(), stacked_bi_lstm_model()]:
        # print("clf:",clf)
        nb_epoch = nb_epoch_h
        batch_size = batch_size_z[j]
        # train_set, test_set = get_stacking(clf, X_train, y_train,X_test)  # second_level_train_set, second_level_test_set
        # print(X_train.shape)    # (10556, 104)

        # print(y_train.shape)     #(10556, 2)
        # print(X_dev.shape)        # (2684, 104)
        train_set, dev_set = get_stacking(clf, X_train, y_train, X_dev)
        train_sets.append(train_set)
        # test_sets.append(test_set)
        dev_sets.append(dev_set)
        j += 1

    meta_train = np.concatenate([result_set.reshape(-1, 2) for result_set in train_sets], axis=1)
    # print("meta_train:", meta_train)
    # meta_test = np.concatenate([y_test_set.reshape(-1, 2) for y_test_set in test_sets], axis=1)
    meta_dev = np.concatenate([y_dev_set.reshape(-1, 2) for y_dev_set in dev_sets], axis=1)
    # print("meta_dev:",meta_dev)

    # 使用决策树作为我们的次级分类器
    from sklearn.tree import DecisionTreeClassifier

    dt_model = DecisionTreeClassifier()
    dt_model.fit(meta_train, y_train)
    # df_predict = dt_model.predict(meta_test)
    df_predict = dt_model.predict(meta_dev)
    # print("df_predict:", df_predict)
    y_pred = np.argmax(df_predict, axis=1)
    # print("y_pred:",y_pred)
    y_dev = np.argmax(y_dev, axis=1)
    # print("y_dev:",y_dev)
    # result_output = pd.DataFrame(data={"id": test["id"], "label": y_pred}, )
    # #
    # result_output.to_csv("./result/task_a_submission_stacking.csv", index=False, quoting=3, )


    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    # print(precision_score(y_dev, df_predict, average='macro'))
    # print(recall_score(y_dev, df_predict, average='macro'))
    # print(accuracy_score(y_dev, df_predict))
    # print(f1_score(y_dev, df_predict, average='macro'))
    print(precision_score(y_dev, y_pred, average='macro'))
    print(recall_score(y_dev, y_pred, average='macro'))
    print(accuracy_score(y_dev, y_pred))
    print(f1_score(y_dev, y_pred, average='macro'))




