#! /usr/bin/env python

import numpy as np
import pandas as pd
from collections import Counter


task_c_train = pd.read_csv("./task_c_train/train.tsv", header = 0, sep = "\t", quoting = 3, )
# task_b_train = pd.read_csv("./train_data/train_subtask_b.tsv", header = 0, sep = "\t", quoting = 3, )
# task_c_train = pd.read_csv("./task_c_train/train.tsv", header = 0, sep = "\t", quoting = 3, )






res1 = task_c_train["subtask_c"]
# res2 = task_c_train["subtask_c"]



def counter(arr):
    """获取每个元素的出现次数，使用标准库collections中的Counter方法"""
    return Counter(arr).most_common(3) # 返回出现频率最高的两个数


def single_list(arr, target):
    """获取单个元素的出现次数，使用list中的count方法"""
    return arr.count(target)


def all_list(arr):
    """获取所有元素的出现次数，使用list中的count方法"""
    result = {}
    for i in set(arr):
        result[i] = arr.count(i)
    return result


def single_np(arr, target):
    """获取单个元素的出现次数，使用Numpy"""
    arr = np.array(arr)
    mask = (arr == target)
    arr_new = arr[mask]
    return arr_new.size


def all_np(arr):
    """获取每个元素的出现次数，使用Numpy"""
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result


if __name__ == "__main__":
    print(counter(res1))
    # print(counter(res2))
    # print(single_list(res2, 2))
    # print(all_list(array))
    # print(single_np(array, 2))
    # print(all_np(array))
