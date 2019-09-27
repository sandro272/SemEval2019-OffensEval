#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/8 20:29

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
from stanford_tokenizer_config import *
from collections import defaultdict


# train = pd.read_csv("../train_data/train_subtask_a", header = 0, sep = "\t", quoting = 3, )
train = pd.read_csv("../train_data/some_train.txt", header = 0, sep = "\t", quoting = 3, )
# test = pd.read_csv("./test_data/test_subtask_a", header = 0, sep = "\t", quoting = 3, )

# print(train.columns)
# print(train.columns)
# print(train.subtask_a)
# print(train.dtypes)

train["subtask_a"] = train["subtask_a"].replace({"NOT":0, "OFF":1})
# train["subtask_b"] = train["subtask_b"].replace({"TIN":0, "UNT":1})
# train["subtask_c"] = train["subtask_c"].replace({"IND":0, "GRP":1, "OTH":2})

# print(train["subtask_a"])

def review_to_wordlist(review_text):

    review_text = spell_correct(review_text)

    review_text = emoji_to_text(review_text)

    review_text = abbreviation_to_text(review_text)

    review_text = emotion_and_split().pre_process_doc(review_text)

    # review_text = " ".join(review_text)     #与review_text = str(review_text)效果一样
    review_text = str(review_text)
    #
    review_text = re.sub("@[\w]*", " ", review_text)

    # review_text = re.sub("[!?,.]", " ", review_text).strip()

    review_text = re.sub("[^a-zA-Z0-9]", " ", review_text)  # 实验3

    # review_text = review_text.lower()

    words = stanford_tokenizer(review_text)

    # return (review_text)
    return(words)

for i in range(len(train["tweet"])):
    # print(train["tweet"][i])                 #输出每一条train["tweet"]
    res = review_to_wordlist(train["tweet"][i])
    print(res)