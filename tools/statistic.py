#! /usr/bin/env python

import numpy as np
import pandas as pd
from collections import Counter
import re
import io

# pd.set_option("display.line_width",1000)
# pd.set_option("display.max_columns",1000)
# pd.set_option("display.max_rows",1000)
# pd.set_option("display.max_colwidth",1000)

# train = pd.read_csv("../train_data/train_subtask_a", header = 0, sep = "\t", quoting = 3,encoding = "utf-8")
# print(train["tweet"])


'''
sentences = ["@USER @USER Go home you’re drunk!!! @USER #MAGA #Trump2020 👊🇺🇸👊 URL",
             "@USER Right on 👍👍  #MAGA 🇺🇸🙏🇺🇸 #WWG1WGA 🇺🇸👊🇺🇸",
             "It is a pleasure see old faces coming back to training and many of you utilising the new timetable 👊🏻👊🏻 ",
             "Corrupt period...😆😆😆👊🏽👊🏽last week u liberals were saying he won’t release bc he has something to hide 😆😆😆😆😆 o boy u pple those pple  that",
             "@USER @USER @USER #SimonSays Jack Viney eats bricks for breakfast 👊  Favourite moment of the game was Melkshams colossal celebration. He is one of our heart and soul players. URL"
            ]


for sentence in sentences:
    print(type(sentence))                            #  <class 'str'>
    if len(sentence) > 0:
        us = re.findall("@USER", sentence)
        res = Counter(us)
        print(res)

'''

'''

def read_file():

    # f = open("../train_data/some_train.txt")
    # readline = f.readlines()

    train = pd.read_csv("../train_data/train_subtask_a", header = 0, sep = "\t", quoting = 3,)
    dic_word = []                        #存储单词

    # 得到文章或者文本的单词并存入列表中

    for tweet in train["tweet"]:
        #因为原文中每个单词都是用空格或者逗号加空格分开的

        # tweet = tweet.replace(",", " ")    #去除逗号，用空格来分开单词
        # tweet = tweet.strip()               #除去左右的空格


        tweet = re.sub("@[\w]*", " ", tweet)

        tweet = re.sub("[a-zA-Z0-9]"," ",tweet)

        tweet = re.sub("[\"\'!?,.]", "", tweet).strip()

        word = tweet.split()

        dic_word.extend(word)

    print(dic_word)
    return dic_word

def clear_count(lists):
    #去除重复的值
    single_word = {}
    single_word = single_word.fromkeys(lists)  #此段代码的意思是将lists的元素作为single_word的键值key    #通过这个代码可以除去重复的列表元素
    word1 = list(single_word.keys())
    #统计单词出现的次数，并存入一个字典中
    for i in word1:
        single_word[i] = lists.count(i)

    return single_word


# clear_count(read_file())


def sort_1(single_word):

    # del[single_word[""]]               #删除' '字符，发现字典中存在空元素，所以删除
                                      #排序，按照values进行排序，如果是按照key进行排序用的d[0]
    single_word_1 = {}
    single_word_1 = sorted(single_word.items(), key = lambda d:d[1], reverse = True)
                                      #得到的是一个列表，里面的元素是元祖，所以再把它转化为字典，不转也可以
    single_word_1 = dict(single_word_1)

    return single_word_1


def main(single_word_1):
    i = 0
    for x,y in single_word_1.items():
        if i < 100:                #输出前20
            print("The word is:","{}".format(x),  ",its amount is:","{}".format(y))
            i += 1
            continue
        else:
            break
    return i


main(sort_1(clear_count(read_file())))

'''

class Counter:
    def __init__(self, path):
        """
        :param path: 文件路径
        """
        self.mapping = dict()
        with io.open(path, encoding="utf-8") as f:
            data = f.read()
            # words = [s.lower() for s in re.findall("\w+\'", data)]
            words = [s for s in re.findall("\w+\'+", data)]
            for word in words:
                self.mapping[word] = self.mapping.get(word, 0) + 1

    def most_common(self, n):
        assert n > 0, "n should be large than 0"
        return sorted(self.mapping.items(), key=lambda item: item[1], reverse=True)[:n]

if __name__ == '__main__':
    # train = pd.read_csv("../train_data/train_subtask_a", header=0, sep="\t", quoting=3, )
    most_common_5 = Counter("../train_data/train_subtask_a.tsv").most_common(1000)
    for item in most_common_5:
        print(item)