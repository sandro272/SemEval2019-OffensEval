#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 19:34
# @Author  : David
# @email   : mingren4792@126.com
# @File    : stacking.py


import numpy as np
import json, argparse, os
import re
import io
import pickle

from sklearn.svm import SVC
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, CuDNNGRU, Bidirectional, GRU, Input, Flatten, SpatialDropout1D, LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import load_model
from config import TWEMOJI_LIST, LOGOGRAM, TWEMOJI, EMOTICONS_TOKEN
from tool import ekphrasis_config
from capsule_net import Capsule
from Attention_layer import AttentionM


# Path to training and testing data file. This data can be downloaded from a link, details of which will be provided.
trainDataPath = ""
testDataPath = ""
# Output file that will be generated. This file can be directly submitted.
solutionPath = ""
# Path to directory where GloVe file is saved.
gloveDir = ""

NUM_FOLDS = None                   # Value of K in K-fold Cross Validation
NUM_CLASSES = None                 # Number of classes - Happy, Sad, Angry, Others
MAX_NB_WORDS = None                # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer
MAX_SEQUENCE_LENGTH = None         # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM = None               # The dimension of the word embeddings
BATCH_SIZE = None                  # The batch size to be chosen for training the model.
LSTM_DIM = None                    # The dimension of the representations learnt by the LSTM model
DROPOUT = None                     # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = None                  # Number of epochs to train a model for



label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '
                line = cSpace.join(lineSplit)

            emoji_repeatedChars = TWEMOJI_LIST
            for emoji_meta in emoji_repeatedChars:
                emoji_lineSplit = line.split(emoji_meta)
                while True:
                    try:
                        emoji_lineSplit.remove('')
                        emoji_lineSplit.remove(' ')
                        emoji_lineSplit.remove('  ')
                        emoji_lineSplit = [x for x in emoji_lineSplit if x != '']
                    except:
                        break
                emoji_cSpace = ' ' + TWEMOJI[emoji_meta][0] + ' '
                line = emoji_cSpace.join(emoji_lineSplit)

            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)

            conv = ' <eos> '.join(line[1:4]) + ' '

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            string = re.sub("tha+nks ", ' thanks ', conv)
            string = re.sub("Tha+nks ", ' Thanks ', string)
            string = re.sub("yes+ ", ' yes ', string)
            string = re.sub("Yes+ ", ' Yes ', string)
            string = re.sub("very+ ", ' very ', string)
            string = re.sub("go+d ", ' good ', string)
            string = re.sub("Very+ ", ' Very ', string)
            string = re.sub("why+ ", ' why ', string)
            string = re.sub("wha+t ", ' what ', string)
            string = re.sub("sil+y ", ' silly ', string)
            string = re.sub("hm+ ", ' hmm ', string)
            string = re.sub("no+ ", ' no ', string)
            string = re.sub("sor+y ", ' sorry ', string)
            string = re.sub("so+ ", ' so ', string)
            string = re.sub("lie+ ", ' lie ', string)
            string = re.sub("okay+ ", ' okay ', string)
            string = re.sub(' lol[a-z]+ ', 'laugh out loud', string)
            string = re.sub(' wow+ ', ' wow ', string)
            string = re.sub('wha+ ', ' what ', string)
            string = re.sub(' ok[a-z]+ ', ' ok ', string)
            string = re.sub(' u+ ', ' you ', string)
            string = re.sub(' wellso+n ', ' well soon ', string)
            string = re.sub(' byy+ ', ' bye ', string)
            string = re.sub(' ok+ ', ' ok ', string)
            string = re.sub('o+h', ' oh ', string)
            string = string.replace('’', '\'').replace('"', ' ').replace("`", "'")
            string = string.replace('fuuuuuuukkkhhhhh', 'fuck')
            string = string.replace('whats ', 'what is ').replace("what's ", 'what is ').replace("i'm ", 'i am ')
            string = string.replace("it's ", 'it is ')
            string = string.replace('Iam ', 'I am ').replace(' iam ', ' i am ').replace(' dnt ', ' do not ')
            string = string.replace('I ve ', 'I have ').replace('I m ', ' I am ').replace('i m ', 'i am ')
            string = string.replace('Iam ', 'I am ').replace('iam ', 'i am ')
            string = string.replace('dont ', 'do not ').replace('google.co.in ', ' google ').replace(' hve ', ' have ')
            string = string.replace(' F ', ' Fuck ').replace('Ain\'t ', ' are not ').replace(' lv ', ' love ')
            string = string.replace(' ok~~ay~~ ', ' okay ').replace(' Its ', ' It is').replace(' its ', ' it is ')
            string = string.replace('  Nd  ', ' and ').replace(' nd ', ' and ').replace('i ll ', 'i will ')
            string = ' ' + string.lower()
            for item in LOGOGRAM.keys():
                string = string.replace(' ' + item + ' ', ' ' + LOGOGRAM[item].lower() + ' ')

            list_str = ekphrasis_config(string)
            for index in range(len(list_str)):
                if list_str[index] in EMOTICONS_TOKEN.keys():
                    list_str[index] = EMOTICONS_TOKEN[list_str[index]][1:len(EMOTICONS_TOKEN[list_str[index]]) - 1].lower()

            for index in range(len(list_str)):
                if list_str[index] in LOGOGRAM.keys():
                    list_str[index] = LOGOGRAM[list_str[index]].lower()

            for index in range(len(list_str)):
                if list_str[index] in LOGOGRAM.keys():
                    list_str[index] = LOGOGRAM[list_str[index]].lower()

            string = ' '.join(list_str)
            indices.append(int(line[0]))
            conversations.append(string.lower())
    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations


def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))

    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def writeNormalisedData(dataFilePath, texts):
    """Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')
                except:
                    # If label information not available (test time)
                    fout.write('\n')


def getElmoEmbeddingMatrix(wordIndex, dim):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(gloveDir, 'elmo.csv'), encoding="utf8") as f:
        for line in f:
            values = line.split('\t')
            word = values[0]
            vec = values[1][1:len(values[1]) - 2]
            vec = vec.split(',')
            embeddingVector = np.asarray(vec, dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, dim))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector

    return embeddingMatrix


def getGloveEmbeddingMatrix(wordIndex, dim):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(gloveDir, 'glove.twitter.27B.200d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, dim))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector

    return embeddingMatrix


def lstmModel(embeddingMatrix, embedding_dim, hidden_dim, name):
    """Constructs the architecture of the modelEMOTICONS_TOKEN[list_str[index]]
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    sequence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               embedding_dim,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(sequence)
    embedded = Dropout(0.3)(embeddingLayer)

    embedded = Bidirectional(LSTM(hidden_dim, dropout=DROPOUT, return_sequences=True))(embedded)
    enc = Bidirectional(LSTM(hidden_dim, dropout=DROPOUT))(embedded)

    fc1 = Dense(128, activation="relu")(enc)
    fc2_dropout = Dropout(0.3)(fc1)

    output = Dense(NUM_CLASSES, activation='sigmoid')(fc2_dropout)
    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model, name


def gruModel(embeddingMatrix, embedding_dim, hidden_dim, name):
    """Constructs the architecture of the modelEMOTICONS_TOKEN[list_str[index]]
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               embedding_dim,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    model = Sequential()
    model.add(embeddingLayer)

    # GRU
    model.add(Bidirectional(GRU(hidden_dim, dropout=DROPOUT, return_sequences=True)))
    model.add(Bidirectional(GRU(hidden_dim, dropout=DROPOUT)))

    model.add(Dense(NUM_CLASSES, activation='sigmoid'))

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model, name


def capsulnetModel(embeddingMatrix,embedding_dim,hidden_dim, name):
    """Constructs the architecture of the modelEMOTICONS_TOKEN[list_str[index]]
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    Routings = 5
    Num_capsule = 10
    Dim_capsule = 32
    embedding_layer = Embedding(embeddingMatrix.shape[0],
                                embedding_dim,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = SpatialDropout1D(0.1)(embedded_sequences)
    x = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))(embedded_sequences)
    x = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True, kernel_size=(3, 1))(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(0.4)(capsule)

    output = Dense(NUM_CLASSES, activation='softmax')(capsule)
    model = Model(inputs=sequence_input, outputs=output)

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model, name


def attentionModel(embeddingMatrix, embedding_dim, hidden_dim, name):
    sequence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               embedding_dim,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)(sequence)
    enc = Bidirectional(GRU(hidden_dim, dropout=DROPOUT, return_sequences=True))(embeddingLayer)
    enc = Bidirectional(GRU(hidden_dim, dropout=DROPOUT, return_sequences=True))(enc)
    att = AttentionM()(enc)
    fc1 = Dense(128, activation="relu")(att)
    fc2_dropout = Dropout(0.25)(fc1)
    output = Dense(4, activation='sigmoid')(fc2_dropout)
    model = Model(inputs=sequence, outputs=output)
    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc'])

    return model, name


def get_stacking(clf, data, labels, x_dev, x_test, n_folds=1, name=None):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num, dev_num = data.shape[0], x_test.shape[0], x_dev.shape[0]
    second_level_train_set = np.zeros((train_num, 4))
    test_result = np.zeros((test_num, 4))
    dev_result = np.zeros((dev_num, 4))
    test_nfolds_sets = []
    dev_nfolds_stes = []

    for k in range(NUM_FOLDS):
        print('-'*80)
        print('Fold %d/%d  %s' %(k+1, NUM_FOLDS, name))
        validationSize = int(len(data)/NUM_FOLDS)
        index1 = validationSize * k
        index2 = validationSize * (k + 1)

        xTrain = np.vstack((data[:index1], data[index2:]))
        yTrain = np.vstack((labels[:index1], labels[index2:]))
        xVal = data[index1:index2]
        yVal = labels[index1:index2]
        print("Building model...")
        early_stopping = EarlyStopping(monitor='val_acc', patience=10)
        clf.fit(xTrain, yTrain, validation_data=[xVal, yVal], epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2,
                callbacks=[early_stopping])
        path = './model/{0}'.format(name)
        if not os.path.exists(path):
            os.makedirs(path)
        clf.save('./model/%s/bi-%s-model-fold-%d.h5' % (name, name, k))

        second_level_train_set[index1:index2] = clf.predict(xVal)
        dev_nfolds_stes.append(clf.predict(x_dev))
        test_nfolds_sets.append(clf.predict(x_test))

    for item in test_nfolds_sets:
        test_result += item
    test_result = test_result / n_folds

    for item in dev_nfolds_stes:
        dev_result += item
    dev_result = dev_result / n_folds

    # second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, dev_result, test_result


def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True, default='testBaseline.config')
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile)

    global trainDataPath, devDataPath, testDataPath, solutionPath, gloveDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE

    trainDataPath = config["train_data_path"]
    devDataPath = config["dev_data_path"]
    testDataPath = config["test_data_path"]

    solutionPath = config["solution_path"]
    gloveDir = config["glove_dir"]

    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    EMBEDDING_DIM = config["embedding_dim"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]
    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]

    print("Processing training data...")
    trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")
    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable
    # writeNormalisedData(trainDataPath, trainTexts)
    print("Processing dev data...")
    devIndices, devTexts = preprocessData(devDataPath, mode="dev")

    print("Processing test data...")
    testIndices, testTexts = preprocessData(testDataPath, mode="test")
    # writeNormalisedData(testDataPath, testTexts)

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(trainTexts)

    # converts text to vector form of word subcript
    trainSequences = tokenizer.texts_to_sequences(trainTexts)
    devSequences = tokenizer.texts_to_sequences(devTexts)
    testSequences = tokenizer.texts_to_sequences(testTexts)
    devData = pad_sequences(devSequences, maxlen=MAX_SEQUENCE_LENGTH)
    testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    pickle.dump(testData, open('./pickle/testData.pickle', 'wb'))

    # save the number for each word, starting at 1
    # #{'some': 1, 'thing': 2,'to': 3 ','eat': 4, drink': 5}
    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    gloveEmbeddingMatrix = getGloveEmbeddingMatrix(wordIndex, dim=200)
    elmoEmbeddingMatrix = getElmoEmbeddingMatrix(wordIndex, dim=1024)

    data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))

    print("Shape of training data tensor: ", data.shape)
    print("Shape of label tensor: ", labels.shape)

    # Randomize data
    # arr = np.arange(len(y_smote_resampled))
    # data = np.array(x_smote_resampled)[arr]
    # labels = np.array(y_smote_resampled)[arr]
    # print(len(data))
    np.random.shuffle(trainIndices)
    data = data[trainIndices]
    labels = labels[trainIndices]

    train_sets = []
    dev_sets = []
    test_sets = []

    print('K-fold start')
    print('-'*60)
    for clf, name in [attentionModel(gloveEmbeddingMatrix, embedding_dim=200, hidden_dim=120, name='gloveAttention'),
                      capsulnetModel(gloveEmbeddingMatrix, embedding_dim=200, hidden_dim=120, name='gloveCapsulnet'),
                      lstmModel(gloveEmbeddingMatrix, embedding_dim=200, hidden_dim=120, name='gloveLstm'),
                      gruModel(gloveEmbeddingMatrix, embedding_dim=200, hidden_dim=120, name='gloveGru'),
                      attentionModel(elmoEmbeddingMatrix, embedding_dim=1024, hidden_dim=400, name='elmoAttention'),
                      capsulnetModel(elmoEmbeddingMatrix, embedding_dim=1024, hidden_dim=400, name='elmoCapsulnet'),
                      lstmModel(elmoEmbeddingMatrix, embedding_dim=1024, hidden_dim=400, name='elmoLstm'),
                      gruModel(elmoEmbeddingMatrix, embedding_dim=1024, hidden_dim=400, name='elmoGru')]:
        train_set, dev_set, test_set = get_stacking(clf, data, labels, devData, testData, name=name)
        train_sets.append(train_set)
        dev_sets.append(dev_set)
        test_sets.append(test_set)

    meta_train = np.concatenate([result_set.reshape(-1, 4) for result_set in train_sets], axis=1)
    meta_dev = np.concatenate([dev_result_set.reshape(-1, 4) for dev_result_set in dev_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1, 4) for y_test_set in test_sets], axis=1)
    path = './pickle/stacking_new_elmo.pickle'
    pickle.dump([meta_train, meta_dev, meta_test, labels], open(path, 'wb'))

    svc = SVC(kernel='sigmoid', gamma=1.3, C=3)
    svc.fit(meta_train, np.array(labels.argmax(axis=1)))
    predictions = svc.predict(meta_test)


    # predictions = predictions.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d"
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))


if __name__ == '__main__':
    main()
