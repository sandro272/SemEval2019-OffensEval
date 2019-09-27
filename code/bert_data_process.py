#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/15 20:13

import re
import pandas as pd
from config import TWEMOJI_LIST, LOGOGRAM, TWEMOJI, EMOTICONS_TOKEN
from preprocess_demo import *
from ekphrasis_tool import ekphrasis_config

train = pd.read_csv("./task_c_train/train.tsv", header=0, sep="\t", quoting=3, )
test = pd.read_csv("./test_data/subtask_test_c.tsv", header = 0, sep = "\t", quoting = 3, )


# train["subtask_b"] = train["subtask_b"].replace({"NOT":0, "OFF":1})
# train["subtask_b"] = train["subtask_b"].replace({"TIN":0, "UNT":1})
train["subtask_c"] = train["subtask_c"].replace({"IND": 0, "GRP": 1, "OTH": 2})

# print(train["subtask_a"])


def review_to_wordlist(review_text):
    repeatedChars = ['.', '?', '!', ',', '"']
    for c in repeatedChars:
        lineSplit = review_text.split(c)
        # print(lineSplit)
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
        review_text = emoji_cSpace.join(emoji_lineSplit)

    review_text = emoji_to_text(review_text)

    review_text = re.sub("(@[\w]*\ )+", " @USER ", review_text)

    duplicateSpacePattern = re.compile(r'\ +')
    review_text = re.sub(duplicateSpacePattern, ' ', review_text).strip()
    # print(review_text)

    string = re.sub("tha+nks ", ' thanks ', review_text)
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


    return (string)


clean_train_reviews = []
for review in train["tweet"]:
    clean_train_reviews.append(review_to_wordlist(review))

clean_test_reviews = []
for review in test["tweet"]:
    clean_test_reviews.append(review_to_wordlist(review))


result_train = pd.DataFrame(data = {"id": train["id"], "tweet": clean_train_reviews, "subtask_c": train["subtask_c"]})
result_train.to_csv("./bert_data_processed/subtask_c/original_train.tsv", index = False, sep = '\t', encoding = "utf-8")

result_test = pd.DataFrame(data = {"id": test["id"], "tweet": clean_test_reviews, "subtask_c": test["subtask_c"]})
result_test.to_csv("./bert_formal/subtask_c/test1.tsv", index = False, sep = '\t', encoding = "utf-8")
