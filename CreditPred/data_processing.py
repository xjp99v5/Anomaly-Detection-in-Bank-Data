#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:06:29 2017

@author: xjp
"""


import pandas as pd
from pandas import DataFrame,Series
import numpy as np
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,scale


#################导入数据####################
filename1 = './data_feature.csv'
data = pd.read_csv(filename1)
############################################



####################填充缺失值####################
mode = 'median'
if mode == 'median':
    data['发生浏览行为总次数'].fillna(data['发生浏览行为总次数'].median(),inplace = True)
    data['不同浏览行为次数'].fillna(data['不同浏览行为次数'].median(),inplace = True)
    data['单种浏览最大次数'].fillna(data['单种浏览最大次数'].median(),inplace = True)
    data['不同浏览子行为次数'].fillna(data['不同浏览子行为次数'].median(),inplace = True)
    data['单种子浏览最大次数'].fillna(data['单种子浏览最大次数'].median(),inplace = True)
    data['单次最大浏览次数'].fillna(data['单次最大浏览次数'].median(),inplace = True)
    data['总交易次数'].fillna(data['总交易次数'].median(),inplace = True)
    data['收入总额'].fillna(data['收入总额'].median(),inplace = True)
    data['支出总额'].fillna(data['支出总额'].median(),inplace = True)
    data['盈亏'].fillna(2,inplace = True)
    data['是否有工资收入'].fillna(2,inplace = True)
    data['银行数量'].fillna(data['银行数量'].median(),inplace = True)
    data['消费笔数'].fillna(data['消费笔数'].median(),inplace = True)
    data['产生循环利息银行数'].fillna(data['产生循环利息银行数'].median(),inplace = True)
    data['逾期总数量'].fillna(data['逾期总数量'].median(),inplace = True)
    data['还款状态为1银行数'].fillna(data['还款状态为1银行数'].median(),inplace = True)
    data['逾期银行数'].fillna(data['逾期银行数'].median(),inplace = True)
elif mode == 'mean':
    data['发生浏览行为总次数'].fillna(data['发生浏览行为总次数'].mean(),inplace = True)
    data['不同浏览行为次数'].fillna(data['不同浏览行为次数'].mean(),inplace = True)
    data['单种浏览最大次数'].fillna(data['单种浏览最大次数'].mean(),inplace = True)
    data['不同浏览子行为次数'].fillna(data['不同浏览子行为次数'].mean(),inplace = True)
    data['单种子浏览最大次数'].fillna(data['单种子浏览最大次数'].mean(),inplace = True)
    data['单次最大浏览次数'].fillna(data['单次最大浏览次数'].mean(),inplace = True)
    data['总交易次数'].fillna(data['总交易次数'].mean(),inplace = True)
    data['收入总额'].fillna(data['收入总额'].mean(),inplace = True)
    data['支出总额'].fillna(data['支出总额'].mean(),inplace = True)
    data['盈亏'].fillna(2,inplace = True)
    data['是否有工资收入'].fillna(2,inplace = True)
    data['银行数量'].fillna(data['银行数量'].mean(),inplace = True)
    data['消费笔数'].fillna(data['消费笔数'].mean(),inplace = True)
    data['产生循环利息银行数'].fillna(data['产生循环利息银行数'].mean(),inplace = True)
    data['逾期总数量'].fillna(data['逾期总数量'].mean(),inplace = True)
    data['还款状态为1银行数'].fillna(data['还款状态为1银行数'].mean(),inplace = True)
    data['逾期银行数'].fillna(data['逾期银行数'].mean(),inplace = True)
else:
    data.dropna(inplace=True)
#################################################


##################删除无用列######################
data.set_index('用户id',inplace=True)
del data['放款时间']
#################################################


###################将类别数据和数值型数据分离，方便后续处理######################
col_category  = ['性别','职业','教育程度','婚姻状态','户口类型','盈亏','是否有工资收入']
col_numerical = ['发生浏览行为总次数','不同浏览行为次数','单种浏览最大次数','不同浏览子行为次数','单种子浏览最大次数',\
                 '单次最大浏览次数','总交易次数','收入总额','支出总额','银行数量','消费笔数','产生循环利息银行数', \
                 '逾期总数量','还款状态为1银行数','逾期银行数']
data_category  = data.loc[:,col_category].copy()
data_numerical = data.loc[:,col_numerical].copy()
#######################处理完毕#########################


##################类别数据做onehot处理###################
data_category  = data_category.astype(object)
enc = OneHotEncoder()
enc.fit(data_category)
data_category_onehot = DataFrame(enc.transform(data_category).toarray())
data_category_onehot.set_index(data_category.index,inplace=True)
####################处理完毕######################


####################数值数据进行归一化###################
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(data_numerical)
data_numerical_minmax = min_max_scaler.transform(data_numerical)
data_numerical_minmax = DataFrame(data_numerical_minmax)
data_numerical_minmax.set_index(data_numerical.index,inplace=True)
#####################处理完毕##########################


####################合并数据并保存#####################
data_final = pd.merge(data_category_onehot,data_numerical_minmax,left_index=True,right_index=True)
data_final.to_csv('data_final.csv',index=None)
#####################################################