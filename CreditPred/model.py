#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:06:29 2017

@author: xjp
"""


import pandas as pd
from pandas import DataFrame, Series
import sklearn
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

#################导入数据##################
filename1 = './data_final.csv'
#filename2 = './overdue_train.csv'
data = pd.read_csv(filename1)
#labels = pd.read_csv(filename2,header=None, names=['用户id', '真实值'])
#########################################


##################建模预测###################
dbscan = DBSCAN(eps=1.5, min_samples=10, metric='euclidean', algorithm='auto', n_jobs=-1)
y_labels = Series(dbscan.fit_predict(data))
params = dbscan.get_params()
###########################################

y_labels[y_labels != 1] = 0
y_labels = y_labels.apply(lambda x : 1-x)
y_labels.to_csv('labels_pre.csv', index=None)

labels = dbscan.labels_  #和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声

print('每个样本的簇标号:')
print(labels)

raito = len(labels[labels[:] == -1]) / len(labels)  #计算噪声点个数占总数的比例
print('噪声比:', format(raito, '.2%'))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

print('分簇的数目: %d' % n_clusters_)
print("轮廓系数: %0.3f" % sklearn.metrics.silhouette_score(data, labels)) #轮廓系数评价聚类的好坏

for i in range(n_clusters_):
    print('簇 ', i, '的所有样本:')
    one_cluster = data[labels == i]
    print(one_cluster)
    plt.plot(one_cluster[:, 0], one_cluster[:, 1], 'o')
