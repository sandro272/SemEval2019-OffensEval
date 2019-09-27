#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/13 12:35

import numpy as np
import pandas as pd
from metrics import f1

test = pd.read_csv("./bert_data_processed/subtask_c/test.tsv",sep="\t",quoting = 3,)
# test = pd.read_csv("./bert_no_process/subtask_c/test.tsv",sep="\t",quoting = 3,)
print(test["subtask_c"])
label_test = test["subtask_c"]

# test = pd.read_csv("./bert_result/label.tsv",sep="\t",quoting = 3,)
# print(test["subtask_a"])
# label_test = test["label"]

result = pd.read_csv("./bert_result/subtask_c/task_c2_result.tsv",sep="\t",quoting = 3,)
# print(result["label"])
label_predict = result["label"]

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

print(precision_score(label_test, label_predict, average='macro'))
print(recall_score(label_test, label_predict, average='macro'))
print(accuracy_score(label_test, label_predict,))
print(f1_score(label_test, label_predict, average='macro'))





# result = csv.reader(open("./result/test_results.tsv", "r+", errors="ignore", encoding="utf-8"), delimiter='\t')
# NOT      # 0
# OFF      # 1
# label = []
# for row in result:
    # print(row[0])      # 第一列
    # print(row[1])      # 第二列
    # NOT.append(row[0])
    # OFF.append(row[1])

    # a = np.argmax(row)
    # label.append(a)
    # print(a)