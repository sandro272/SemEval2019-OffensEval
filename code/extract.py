#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/12 11:40

import pandas as pd


# task_b_train = pd.read_csv("./train_data/train_subtask_b.tsv", header = 0, sep = "\t", quoting = 3, )
task_c_train = pd.read_csv("./train_data/train_subtask_c.tsv", header = 0, sep = "\t", quoting = 3, )
# print(task_b_train)

# task_b_train["subtask_b"] == "TIN" or task_b_train["subtask_b"] == "UNT"

# task_b_train = task_b_train.dropna(axis = 0)     # axis = 0 删除行
task_c_train = task_c_train.dropna(axis = 0)

# print(res)
# print(task_b_train["subtask_b"].isnull().sum())      # 8840
# print(len(res))                                      # 4400

# task_b_train = pd.DataFrame(data = {"id": task_b_train["id"], "tweet": task_b_train["tweet"], "subtask_b": task_b_train["subtask_b"]})
# task_b_train.to_csv("./task_b_train/train.tsv", index = False, sep = '\t', encoding = "utf-8")

task_c_train = pd.DataFrame(data = {"id": task_c_train["id"], "tweet": task_c_train["tweet"], "subtask_c": task_c_train["subtask_c"]})
task_c_train.to_csv("./task_c_train/train.tsv", index = False, sep = '\t', encoding = "utf-8")





