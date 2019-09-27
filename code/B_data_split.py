#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/20 20:19

import pandas as pd




#  划分 train_data, dev_data

df = pd.read_csv("./bert_data_processed/subtask_b/original_train.tsv", header = 0, sep = "\t", quoting = 3, )
# df["subtask_b"] = df["subtask_b"].replace({"TIN": 0, "UNT": 1})

df = df.sample(frac=1.0)  # 全部打乱
# print(df.shape[0])     #   4437

cut_idx1 = int(round(0.1 * df.shape[0]))

df_dev, df_train = df.iloc[:cut_idx1], df.iloc[cut_idx1:]

split_train = pd.DataFrame(data = {"id": df_train["id"], "tweet": df_train["tweet"], "subtask_b": df_train["subtask_b"]})
split_train.to_csv("./bert_data_processed/subtask_b/gd_train.tsv", index = False, sep = '\t', encoding = "utf-8")



split_dev = pd.DataFrame(data = {"id": df_dev["id"], "tweet": df_dev["tweet"], "subtask_b": df_dev["subtask_b"]})
split_dev.to_csv("./bert_data_processed/subtask_b/dev.tsv", index = False, sep = '\t', encoding = "utf-8")


'''


# 划分 train_data, test_data

df = pd.read_csv("./bert_data_processed/subtask_b/gd_train.tsv", header = 0, sep = "\t", quoting = 3, )
df = df.sample(frac=1.0)  # 全部打乱
# print(df.shape[0])                                 # 13240
cut_idx1 = int(round(0.1 * df.shape[0]))
df_test, df_train = df.iloc[:cut_idx1], df.iloc[cut_idx1:]    #iloc按位置进行提取
print(df_test.shape)            #  (1192, 3)
print(df_train.shape)           #  (10724, 3)


split_train = pd.DataFrame(data = {"id": df_train["id"], "tweet": df_train["tweet"], "subtask_b": df_train["subtask_b"]})
split_train.to_csv("./bert_data_processed/subtask_b/train.tsv", index = False, sep = '\t', encoding = "utf-8")


# split_test = pd.DataFrame(data = {"id": df_test["id"], "tweet": df_test["tweet"]})
# split_test.to_csv("./bert_data_processed/nolabel_test.tsv", index = False, sep = '\t', encoding = "utf-8")
#
# label = pd.DataFrame(data={"id": df_test["id"], "label": df_test["subtask_a"]})
# label.to_csv("./bert_data_processed/label.tsv",  index = False, sep = '\t', encoding = "utf-8")

split_test = pd.DataFrame(data = {"id": df_test["id"], "tweet": df_test["tweet"], "subtask_b": df_test["subtask_b"]})
split_test.to_csv("./bert_data_processed/subtask_b/test.tsv", index = False, sep = '\t', encoding = "utf-8")

'''
