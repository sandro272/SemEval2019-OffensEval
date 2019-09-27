#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/10 13:30

import pandas as pd


df = pd.read_csv("./train_data/train_subtask_a.tsv", header = 0, sep = "\t", quoting = 3, )
# df["subtask_a"] = df["subtask_a"].replace({"NOT":0, "OFF":1})
#  划分 train_dta, dev_data





# df = pd.read_csv("./bert_formal/original_train.tsv", header = 0, sep = "\t", quoting = 3, )
# df["subtask_a"] =  df["subtask_a"].replace({"NOT":0, "OFF":1})
df = df.sample(frac=1.0)  # 全部打乱
# print(df.shape[0])                                 # 13240
cut_idx1 = int(round(0.1 * df.shape[0]))
df_dev, df_train = df.iloc[:cut_idx1], df.iloc[cut_idx1:]

split_train = pd.DataFrame(data = {"id": df_train["id"], "tweet": df_train["tweet"], "subtask_a": df_train["subtask_a"]})
split_train.to_csv("./data/gd_train.tsv", index = False, sep = '\t', encoding = "utf-8")


split_dev = pd.DataFrame(data = {"id": df_dev["id"], "tweet": df_dev["tweet"]})
split_dev.to_csv("./data/dev.tsv", index = False, sep = '\t', encoding = "utf-8")

# print(df["subtask_a"])

# print(df.shape)                     #(13240, 3)
# print(df_dev.shape)                   #(1324, 3)
# print(df_train.shape)                  #(11916, 3)
'''


# 划分 train_data, test_data

df = pd.read_csv("./bert_no_process/among_train.tsv", header = 0, sep = "\t", quoting = 3, )
df = df.sample(frac=1.0)  # 全部打乱
# print(df.shape[0])                                 # 13240
cut_idx1 = int(round(0.1 * df.shape[0]))
df_test, df_train = df.iloc[:cut_idx1], df.iloc[cut_idx1:]    #iloc按位置进行提取
# print(df_test.shape)            #  (1192, 3)
# print(df_train.shape)           #  (10724, 3)


split_train = pd.DataFrame(data = {"id": df_train["id"], "tweet": df_train["tweet"], "subtask_a": df_train["subtask_a"]})
split_train.to_csv("./bert_no_process/label_train.tsv", index = False, sep = '\t', encoding = "utf-8")


# split_test = pd.DataFrame(data = {"id": df_test["id"], "tweet": df_test["tweet"]})
# split_test.to_csv("./bert_data_processed/nolabel_test.tsv", index = False, sep = '\t', encoding = "utf-8")
#
# label = pd.DataFrame(data={"id": df_test["id"], "label": df_test["subtask_a"]})
# label.to_csv("./bert_data_processed/label.tsv",  index = False, sep = '\t', encoding = "utf-8")

split_test = pd.DataFrame(data = {"id": df_test["id"], "tweet": df_test["tweet"], "subtask_a": df_test["subtask_a"]})
split_test.to_csv("./bert_no_process/label_test.tsv", index = False, sep = '\t', encoding = "utf-8")

'''
