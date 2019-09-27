#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/5 10:31
# @Author  : David
# @email   : mingren4792@126.com
# @File    : demo.py

import pandas as pd

train = pd.read_csv('./data/version_1/train_merge.csv', sep='\t')
# data = pd.read_csv('./demo.csv', )
# print(data)

i = 0;
result = []
with open('./demo.csv', 'rb') as f:
    for line in f:
        if len(line) - len(train['review'][i]) < 0:
            print(i)
            print(train['review'][i])
            print(line)
            print('------------')
            break;
        i += 1

