"""
Example from training to saving.
"""
import argparse
import os
import re
import numpy as np
import io
import pandas as pd

# from anago.utils import load_data_and_labels, load_glove, filter_embeddings
from anago_preprocessing import ELMoTransformer, IndexTransformer

from config import TWEMOJI_LIST, LOGOGRAM, TWEMOJI, EMOTICONS_TOKEN
from tool import ekphrasis_config

from preprocess_demo import *
from collections import defaultdict



label2emotion = {0: "IND", 1: "GRP", 2: "OTH"}
emotion2label = {"IND": 0, "GRP": 1, "OTH": 2}


# data
def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn(list cut?) conversations, processed and each turn separated by the <eos> tag
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
            repeatedChars = ['.', '?', '!', ',', '"']
            for c in repeatedChars:
                lineSplit = line.split(c)
                # print(lineSplit)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '
                line = cSpace.join(lineSplit)
                # print(lineSplit)

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
                # print(line)

            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[2]]
                # print(label)
                labels.append(label)


            # conv = ' <eos> '.join(line[1:4]) + ' '
            conv = ' <eos> '.join(line[1:2]) + ' '
            # print(conv)
            # print(line[0])    # id
            # print(line[1])    # tweet
            # print(line[2])      # lable

            conv = emoji_to_text(conv)

            conv = re.sub("(@[\w]*\ )+", " @USER ", conv)

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', str(conv))

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
            review_text = re.sub(' byy+ ', ' bye ', string)
            # review_text = re.sub("(im\s)+", " i am ", review_text)
            review_text = re.sub("(\wl\ss\w)+", ' also ', review_text)
            # review_text = re.sub("(IM\s)+", " i am ", review_text)
            review_text = re.sub("(\sbro$)+", " brother ", review_text)
            review_text = re.sub("\stv", " Television ", review_text)
            # review_text = review_text.replace('’', '\'').replace('"', ' ').replace("`", "'")

            review_text = abbreviation_to_text(review_text)

            string = review_text.replace('whats ', 'what is ').replace(" i'm ", 'i am ')
            string = string.replace("it's ", 'it is ')
            string = string.replace('Iam ', 'I am ').replace(' iam ', ' i am ').replace(' dnt ', ' do not ')
            string = string.replace('I ve ', 'I have ').replace(' I m ', ' I\'am ').replace(' i m ', 'i\'m ')
            string = string.replace(' Iam ', 'I am ').replace(' iam ', 'i am ')
            string = string.replace('dont ', 'do not ').replace('google.co.in ', ' google ').replace(' hve ', ' have ')
            string = string.replace(' F ', ' Fuck ').replace('Ain\'t ', ' are not ').replace(' lv ', ' love ')
            string = string.replace(' ok~~ay~~ ', ' okay ').replace(' Its ', ' It is').replace(' its ', ' it is ')
            string = string.replace('  Nd  ', ' and ').replace(' nd ', ' and ').replace('i ll ', 'i will ')
            string = ' ' + string.lower()
            indices.append(int(line[0]))
            conversations.append(string.lower())
    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations


# 获取数据集词典
def get_word_dict():
    trainDataPath = './elmo_data/gd_train.txt'
    devDataPath = './elmo_data/dev.txt'
    testDataPath = './elmo_data/test.txt'
    word_map = defaultdict(float)
    trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")
    print("trainIndices:",trainIndices)
    print("trainTexts:", trainTexts)
    print("labels:",labels)
    devIndices, devTexts = preprocessData(devDataPath, mode="dev")
    testIndices, testTexts = preprocessData(testDataPath, mode="test")
    for train_item in trainTexts:
        train_text_list = train_item.split()
        for train_word in train_text_list:
            word_map[train_word] += 1
    
    for test_item in testTexts:
        test_text_list = test_item.split()
        for test_word in test_text_list:
            word_map[test_word] += 1
    
    for dev_item in devTexts:
        dev_text_list = dev_item.split()
        for test_word in dev_text_list:
            word_map[test_word] += 1

    return word_map


# res = get_word_dict()
# print(res)

 # 获取数据集word
def elmo_feature():
    word_map = get_word_dict()
    result = []
    print('Transforming datasets...')
    p = ELMoTransformer()
    i = 0
    #  遍历词字典，生成对应向量
    for item in word_map.keys():
        i += 1
        print(i)
        # print(item)
        vec = []
        #  格式要求，我没有修改
        train = [[item+'']]
        # 获取转换器，预训练开始
        word_vec = p.transform(train)
        for m in word_vec:
            for n in m:
                word_vec = n
        # print("word_vec:",word_vec)
        vec.append(item)
        #  返回值是四层数组
        # meta_vec = word_vec[0][0][0]
        meta_vec = word_vec

        # print('hhhhhhhhhhhhhhhh')
        # print(meta_vec)
        meta_vec = meta_vec.tolist()
        vec.append(meta_vec)
        # print("vec:",vec)
        result.append(vec)
    df = pd.DataFrame(data=result)
    df.to_csv('./elmo_data/elmo_terminal.csv', sep='\t', index=False, header=None, encoding='utf-8')


def char_featur():
    word_map = get_word_dict()
    # print(word_map.keys())
    result = []

    c = IndexTransformer()
    for item in word_map.keys():
        # print(item)
        vec = []
        train = [[item + '']]
        # print(train)
        word_vec = c.transform(train)

        print(word_vec)
        vec.append(item)
        # meta_vec = word_vec[0]
        meta_vec = word_vec[0][0][0]
        meta_vec = meta_vec.tolist()
        vec.append(meta_vec)
        result.append(vec)
        # print(result)





if __name__ == '__main__':
    elmo_feature()
    # char_featur()
