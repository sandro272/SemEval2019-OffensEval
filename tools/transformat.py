#! /usr/bin/env python

import pandas as pd

train = pd.read_csv("./train_data/offenseval-training-v1.tsv", header = 0, sep = "\t", quoting = 3, )

# train["subtask_a"] = train["subtask_a"].replace({"NOT":0, "OFF":1})
# train["subtask_b"] = train["subtask_b"].replace({"TIN":0, "UNT":1})
# train["subtask_c"] = train["subtask_c"].replace({"IND":0, "GRP":1, "OTH":2})

# print(train["id"])
# print(train["subtask_a"])

train_subtask_a = pd.DataFrame(data = {"id": train["id"], "tweet": train["tweet"], "subtask_a": train["subtask_a"]})
train_subtask_a.to_csv("../train_data/train_subtask_a", index = False, sep = '\t', quotechar = '\'', encoding = "utf-8")



train_subtask_b = pd.DataFrame(data = {"id": train["id"], "tweet": train["tweet"], "subtask_b": train["subtask_b"]})
train_subtask_b.to_csv("../train_data/train_subtask_b", index = False, sep = '\t', quotechar = '\'', encoding = "utf-8")


train_subtask_c = pd.DataFrame(data = {"id": train["id"], "tweet": train["tweet"], "subtask_c": train["subtask_c"]})
train_subtask_c.to_csv("../train_data/train_subtask_c", index = False, sep = '\t', quotechar = '\'', encoding = "utf-8")

