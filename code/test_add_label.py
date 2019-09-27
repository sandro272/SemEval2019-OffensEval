#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/18 2:23

import pandas as pd


test = pd.read_csv("./test_data/test_set_taskc.tsv", header = 0, sep = "\t", quoting = 3, )


split_test = pd.DataFrame(data = {"id": test["id"], "tweet": test["tweet"], "subtask_c": "0"})
split_test.to_csv("./test_data/subtask_test_c.tsv", index = False, sep = '\t', encoding = "utf-8")
