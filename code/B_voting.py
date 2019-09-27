#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/24 8:44

import os
import sys
import logging
import pandas as pd
import numpy as np
import pickle

from util import save_result
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers import Dense, Input, SpatialDropout1D, CuDNNGRU, Dropout, MaxPooling2D, Flatten, LSTM, MaxPooling1D, Embedding, Bidirectional, GRU
from keras.models import Model
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from Attention_layer import AttentionM
from capsule_net import Capsule
from vote_classifier import VotingClassifier
from metrics import f1
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold #分层k折交叉验证
from keras.callbacks import ModelCheckpoint,EarlyStopping



train = pd.read_csv("./task_b_train/train1.tsv", header=0, sep="\t", quoting=3, )
test = pd.read_csv("./test_data/testset-taskb.tsv", header=0, sep="\t", quoting=3, )


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


def make_idx_data(revs, word_idx_map, maxlen=120):
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

    return [X_train, X_test, X_dev, y_train, y_dev,]


def bi_lstm_model(batch_size, nb_epoch, hidden_dim, num):


    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)

    embedded = Dropout(0.25)(embedded)

    # bi-directional LSTM
    hidden = Bidirectional(LSTM(hidden_dim//2, recurrent_dropout=0.25)) (embedded)


    output = Dense(2, activation='softmax')(hidden)
    model = Model(inputs=sequence, outputs=output)
    class_weight = {0: 1, 1: 7}

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])

    # train_num, test_num = X_train.shape[0], X_dev.shape[0]
    train_num, test_num = X_train.shape[0], X_test.shape[0]
    num1 = y_train.shape[1]

    second_level_train_set = np.zeros((train_num, num1))

    second_level_test_set = np.zeros((test_num, num1))

    test_nfolds_sets = []



    # kf = KFold(n_splits = 2)
    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train)):

        x_tra, y_tra = X_train[train_index], y_train[train_index]

        x_tst, y_tst = X_train[test_index], y_train[test_index]

        # checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)

        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight, callbacks=[early_stopping])


        second_level_train_set[test_index] = model.predict(x_tst,batch_size=batch_size)

        test_nfolds_sets.append(model.predict(X_test))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    model.save("weights_BB_bi_lstm" + num + ".hdf5")

    y_pred = second_level_test_set

    return y_pred


def bi_gru_model(batch_size, nb_epoch, hidden_dim, num):
    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)

    embedded = Dropout(0.25)(embedded)


    # bi-directional GRU
    hidden = Bidirectional(GRU(hidden_dim // 2, recurrent_dropout=0.25))(embedded)

    output = Dense(2, activation='softmax')(hidden)
    model = Model(inputs=sequence, outputs=output)
    class_weight = {0: 1, 1: 7}

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])

    # train_num, test_num = X_train.shape[0], X_dev.shape[0]
    train_num, test_num = X_train.shape[0], X_test.shape[0]
    num1 = y_train.shape[1]


    second_level_train_set = np.zeros((train_num, num1))

    second_level_test_set = np.zeros((test_num, num1))  # (2684,)

    test_nfolds_sets = []

    # kf = KFold(n_splits=5)
    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train)):

        x_tra, y_tra = X_train[train_index], y_train[train_index]

        x_tst, y_tst = X_train[test_index], y_train[test_index]


        # checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_acc', patience=12, verbose=1)

        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight, callbacks=[early_stopping])

        second_level_train_set[test_index] = model.predict(x_tst, batch_size=batch_size)

        test_nfolds_sets.append(model.predict(X_test))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    model.save("weights_BB_bi_gru" + num + ".hdf5")

    y_pred = second_level_test_set

    return y_pred

def attention_lstm_model(batch_size, nb_epoch, hidden_dim, num):
    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)

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

    # train_num, test_num = X_train.shape[0], X_dev.shape[0]
    train_num, test_num = X_train.shape[0], X_test.shape[0]
    num1 = y_train.shape[1]

    second_level_train_set = np.zeros((train_num, num1))  # (10556,)

    second_level_test_set = np.zeros((test_num, num1))  # (2684,)

    test_nfolds_sets = []

    # kf = KFold(n_splits=5)
    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        x_tra, y_tra = X_train[train_index], y_train[train_index]

        x_tst, y_tst = X_train[test_index], y_train[test_index]

        # checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)

        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight, callbacks=[early_stopping])

        second_level_train_set[test_index] = model.predict(x_tst, batch_size=batch_size)

        test_nfolds_sets.append(model.predict(X_test))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    model.save("weights_BB_attention_lstm" + num + ".hdf5")

    y_pred = second_level_test_set

    return y_pred


def capsulnet_model(batch_size, nb_epoch, hidden_dim, num):
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
    # checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    # early_stopping = EarlyStopping(monitor='val_acc', patience = 5, verbose=1)
    class_weight = {0: 1, 1: 7}

    # train_num, test_num = X_train.shape[0], X_dev.shape[0]
    train_num, test_num = X_train.shape[0], X_test.shape[0]
    num1 = y_train.shape[1]

    second_level_train_set = np.zeros((train_num, num1))  # (10556,)

    second_level_test_set = np.zeros((test_num, num1))  # (2684,)

    test_nfolds_sets = []

    # kf = KFold(n_splits=5)
    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        x_tra, y_tra = X_train[train_index], y_train[train_index]

        x_tst, y_tst = X_train[test_index], y_train[test_index]

        # checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_acc', patience=8, verbose=1)

        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight, callbacks=[early_stopping])

        second_level_train_set[test_index] = model.predict(x_tst,
                                                           batch_size=batch_size)  # (2112,2) could not be broadcast to indexing result of shape (2112,)

        test_nfolds_sets.append(model.predict(X_test))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    model.save("weights_BB_capsulnet_lstm" + num + ".hdf5")

    y_pred = second_level_test_set

    return y_pred

def stacked_lstm_model(batch_size, nb_epoch, hidden_dim, num):
    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
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
    # checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    # early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1)

    class_weight = {0: 1, 1: 7}

    # train_num, test_num = X_train.shape[0], X_dev.shape[0]
    train_num, test_num = X_train.shape[0], X_test.shape[0]
    num1 = y_train.shape[1]

    second_level_train_set = np.zeros((train_num, num1))  # (10556,)

    second_level_test_set = np.zeros((test_num, num1))  # (2684,)

    test_nfolds_sets = []

    # kf = KFold(n_splits=5)
    kf = KFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        x_tra, y_tra = X_train[train_index], y_train[train_index]

        x_tst, y_tst = X_train[test_index], y_train[test_index]

        # checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1)

        model.fit(x_tra, y_tra, validation_data=[x_tst, y_tst], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  class_weight=class_weight, callbacks=[early_stopping])

        second_level_train_set[test_index] = model.predict(x_tst,
                                                           batch_size=batch_size)  # (2112,2) could not be broadcast to indexing result of shape (2112,)

        test_nfolds_sets.append(model.predict(X_test))
    for item in test_nfolds_sets:
        second_level_test_set += item

    second_level_test_set = second_level_test_set / 5

    model.save("weights_BB_stacked_lstm" + num + ".hdf5")

    y_pred = second_level_test_set

    return y_pred


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    # pickle_file = sys.argv[1]
    pickle_file = os.path.join('pickle', 'SemEval_train_val_test_TASK_B_terminal.pickle3')

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

    bi_lstm_pre1 = bi_lstm_model(256, 40, 120, '1')
    bi_lstm_pre2 = bi_lstm_model(256, 40, 120, '2')
    bi_lstm_pre3 = bi_lstm_model(256, 40, 120, '3')
    bi_lstm_pre4 = bi_lstm_model(256, 40, 120, '4')
    bi_lstm_pre5 = bi_lstm_model(256, 40, 120, '5')
    bi_lstm_pre = (bi_lstm_pre1 + bi_lstm_pre2 + bi_lstm_pre3 + bi_lstm_pre4 + bi_lstm_pre5) / 5
    # bi_lstm_pre = bi_lstm_pre1

    bi_gru_pre1 = bi_gru_model(100, 40, 120, '1')
    bi_gru_pre2 = bi_gru_model(100, 40, 120, '2')
    bi_gru_pre3 = bi_gru_model(100, 40, 120, '3')
    bi_gru_pre4 = bi_gru_model(100, 40, 120, '4')
    bi_gru_pre5 = bi_gru_model(100, 40, 120, '5')
    bi_gru_pre = (bi_gru_pre1 + bi_gru_pre2 + bi_gru_pre3 + bi_gru_pre4 + bi_gru_pre5) / 5
    # bi_gru_pre = bi_gru_pre1

    attention_lstm_pre1 = attention_lstm_model(256, 40, 120, '1')
    attention_lstm_pre2 = attention_lstm_model(256, 40, 120, '2')
    attention_lstm_pre3 = attention_lstm_model(256, 40, 120, '3')
    attention_lstm_pre4 = attention_lstm_model(256, 40, 120, '4')
    attention_lstm_pre5 = attention_lstm_model(256, 40, 120, '5')
    attention_lstm_pre = (attention_lstm_pre1 + attention_lstm_pre2 + attention_lstm_pre3 + attention_lstm_pre4 + attention_lstm_pre5) / 5
    # attention_lstm_pre = attention_lstm_pre1

    capsulnet_pre1 = capsulnet_model(100, 40, 120, "1")
    capsulnet_pre2 = capsulnet_model(100, 40, 120, "2")
    capsulnet_pre3 = capsulnet_model(100, 40, 120, "3")
    capsulnet_pre4 = capsulnet_model(100, 40, 120, "4")
    capsulnet_pre5 = capsulnet_model(100, 40, 120, "5")
    capsulnet_pre = (capsulnet_pre1 + capsulnet_pre2 + capsulnet_pre3 + capsulnet_pre4 + capsulnet_pre5) / 5
    # capsulnet_pre = capsulnet_pre1

    stacked_pre1 = stacked_lstm_model(100, 40, 120, '1')
    stacked_pre2 = stacked_lstm_model(100, 40, 120, '2')
    stacked_pre3 = stacked_lstm_model(100, 40, 120, '3')
    stacked_pre4 = stacked_lstm_model(100, 40, 120, '4')
    stacked_pre5 = stacked_lstm_model(100, 40, 120, '5')
    stacked_pre = (stacked_pre1 + stacked_pre2 + stacked_pre3 + stacked_pre4 + stacked_pre5) / 5
    # stacked_pre = stacked_pre1

    y_pred = (bi_lstm_pre + bi_gru_pre + attention_lstm_pre + capsulnet_pre + stacked_pre)
    y_pred = np.argmax(y_pred, axis=1)
    print("y_pred:",y_pred)
    # print(len(y_pred))

    result_output = pd.DataFrame(data={"id": test["id"], "label": y_pred}, )

    result_output.to_csv("./result/subtask_b/task_BB_submission_voting.csv", index=False, quoting=3, )


