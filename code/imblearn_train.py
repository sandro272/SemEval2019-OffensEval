#! /usr/bin/env python
# _*_coding:utf-8_*_
# project: SemEval2019
# Author: zcj
# @Time: 2019/1/19 23:45


from collections import Counter
from sklearn.model_selection import train_test_split  #数据集划分
import pandas as pd
import numpy as np
train = pd.read_csv("./task_b_train/train1.tsv", header=0, sep="\t", quoting=3, )

train["subtask_b"] = train["subtask_b"].replace({"TIN": 0, "UNT": 1})

'''
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0) #采用随机过采样（上采样）
x_resample,y_resample=ros.fit_sample(train, train["subtask_b"])
# print(len(x_resample))
# print(len(y_resample))

id = []
label = []
tweet = []

for x_item in x_resample:
    # print(x_item[0])
    id.append(x_item[0])
    label.append(x_item[1]) 
    tweet.append(x_item[2])
    # print(x_item[2])
    
    
train_b = pd.DataFrame({"id":id, "tweet":tweet, "subtask_b":label})
train_b.to_csv("./task_b_train/train2.tsv", index = False, sep = '\t', encoding = "utf-8")

# 出现过拟合现象 0.92 ---0.95
'''

'''
from sklearn.model_selection import StratifiedKFold #分层k折交叉验证
import numpy as np
kf = StratifiedKFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(train["tweet"], train["subtask_b"]):

    x_train = np.array(train["tweet"])[train_index]
    y_train = np.array(train["tweet"])[train_index]
    x_test = np.array(train["tweet"])[test_index]
    y_test = np.array(train["tweet"])[test_index]

# print(len(train_index))     # 3994
# print(len(test_index))     #  443

# print(x_train.shape)    #  （3994，）
print(y_train.shape)            #  (3994,)
# print(x_test)
# print(y_test)

'''

import numpy as np
import keras
a = np.array([[1,2],
          [4,5]])
print(a)
print(a.shape)

b = keras.utils.to_categorical(a)
print(b.shape)





