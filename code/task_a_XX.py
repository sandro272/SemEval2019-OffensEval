#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/18 4:43

import pandas as pd


# test = pd.read_csv("./bert_formal/subtask_c/test1.tsv",sep="\t",quoting = 3,)
result = pd.read_csv("./result/subtask_c/task_C_submission_voting.csv", sep=",", quoting = 3, encoding = "utf-8")
# print(result["label"])


pre = pd.DataFrame(data = {"id": result["id"], "label": result["label"].replace({0: "IND", 1: "GRP", 2: "OTH"})})
pre.to_csv("./result/subtask_c/subtask_C_submission_voting_terminal.csv", index = False, encoding = "utf-8",header = 0)