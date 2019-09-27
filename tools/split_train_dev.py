#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/10 13:30

import pandas as pd


'''
df = pd.read_csv("./train_data/train_subtask_a", header = 0, sep = "\t", quoting = 3, )
df = df.sample(frac=1.0)  # 全部打乱
# print(df.shape[0])                                 # 13240
cut_idx1 = int(round(0.1 * df.shape[0]))
df_dev, df_train = df.iloc[:cut_idx1], df.iloc[cut_idx1:]

# print(df.shape)                     #(13240, 3)
# print(df_dev.shape)                   #(1324, 3)
# print(df_train.shape)                  #(11916, 3)
'''


df = pd.read_csv("./train_data/split_train", header = 0, sep = "\t", quoting = 3, )
df = df.sample(frac=1.0)  # 全部打乱
# print(df.shape[0])                                 # 13240
cut_idx1 = int(round(0.1 * df.shape[0]))
df_test, df_train = df.iloc[:cut_idx1], df.iloc[cut_idx1:]
# print(df_test.shape)            #  (1192, 3)
# print(df_train.shape)           #  (10724, 3)


# split_dev = pd.DataFrame(data = {"id": df_dev["id"], "tweet": df_dev["tweet"], "subtask_a": df_dev["subtask_a"]})
# split_dev.to_csv("./train_data/split_dev", index = False, sep = '\t', quotechar = '\'', encoding = "utf-8")


# split_test = pd.DataFrame(data = {"id": df_test["id"], "tweet": df_test["tweet"], "subtask_a": df_test["subtask_a"]})
# split_test.to_csv("./train_data/split_test", index = False, sep = '\t', quotechar = '\'', encoding = "utf-8")
#
# split_train = pd.DataFrame(data = {"id": df_train["id"], "tweet": df_train["tweet"], "subtask_a": df_train["subtask_a"]})
# split_train.to_csv("./train_data/split_train1", index = False, sep = '\t', quotechar = '\'', encoding = "utf-8")