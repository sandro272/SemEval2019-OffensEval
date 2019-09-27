#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/17 3:35
import pandas as pd


# res = pd.read_csv("./result/task_a_submission_voting.csv",header = 0, quoting = 3, )
# # res = res["label"].replace({0:"NOT", 1:"OFF"})
#
#
# result_output = pd.DataFrame(data={"id": res["id"], "label": res["label"].replace({0:"NOT", 1:"OFF"})})
# result_output.to_csv("./result/subtask_a_submission_voting.csv", index=False, quoting=3, header = 0, encoding = "utf-8")

res = pd.read_csv("./result/subtask_b/B_stacked_result.csv", header = 0, quoting = 3, )
# res = res["label"].replace({0:"NOT", 1:"OFF"})


result_output = pd.DataFrame(data={"id": res["id"], "label": res["label"].replace({0:"TIN", 1:"UNT"})})
result_output.to_csv("./result/subtask_b/stacked_bi_lstm_submission.csv", index=False, quoting=3, header = 0, encoding = "utf-8")