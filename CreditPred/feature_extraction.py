#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:06:29 2017

@author: xjp
"""

import pandas as pd
from pandas import DataFrame,Series
import numpy as np

#############导入数据##############
filename1 = './bank_detail_train.csv'
filename2 = './bill_detail_train.csv'
filename3 = './browse_history_train.csv'
filename4 = './loan_time_train.csv'
filename5 = './overdue_train.csv'
filename6 = './user_info_train.csv'
bank_detail_train    = pd.read_csv(filename1,header=None,names=['用户id','时间戳','交易类型','交易金额','工资收入标记'])
bill_detail_train    = pd.read_csv(filename2,header=None,names=['用户id','账单时间戳','银行id','上期账单金额','上期还款金额',\
                                                                '信用卡额度','本期账单余额','本期账单最低还款额','消费笔数',\
                                                                '本期账单金额','调整金额','循环利息','可用金额','预借现金额度',\
                                                                '还款状态'])
browse_history_train = pd.read_csv(filename3,header=None,names=['用户id','时间戳','浏览行为数据','浏览子行为编号'])
loan_time_train      = pd.read_csv(filename4,header=None,names=['用户id','放款时间'])
overdue_train        = pd.read_csv(filename5,header=None,names=['用户id','样本标签'])
user_info_train      = pd.read_csv(filename6,header=None,names=['用户id','性别','职业','教育程度','婚姻状态','户口类型'])

#######合并user_info,loan_time,将用户静态数据保存以待建模########
data_user_info = pd.merge(user_info_train,loan_time_train,on='用户id')
data_user_info.sort_values(by='用户id',ascending=True,inplace=True)



#######从browse_history_train表中获取特征#########
#######1.获取用户浏览行为总次数数据#################
grouped_browse_1 = browse_history_train[['用户id','浏览行为数据']].groupby(['用户id'])
browse_times = grouped_browse_1.count()
browse_times.reset_index(inplace=True)
browse_times.columns = ['用户id','发生浏览行为总次数']
#######2.获取用户不同浏览行为次数数据###############
browse_times_unique = {}
browse_times_unique_max = {}
for name,group in grouped_browse_1:
    browse_times_unique[name] = len(group['浏览行为数据'].unique())
    browse_times_unique_max[name]=max(group['浏览行为数据'].value_counts())
browse_times_unique=DataFrame(Series(browse_times_unique))
browse_times_unique.reset_index(inplace=True)
browse_times_unique.columns = ['用户id','不同浏览行为次数']
#######3.获取用户单种浏览行为最大次数###############
browse_times_unique_max=DataFrame(Series(browse_times_unique_max))
browse_times_unique_max.reset_index(inplace=True)
browse_times_unique_max.columns = ['用户id','单种浏览最大次数']
#######4.获取用户不同浏览子行为次数数据#############
grouped_browse_2 = browse_history_train[['用户id','浏览子行为编号']].groupby(['用户id'])
browse_times_unique_ch = {}
browse_times_unique_max_ch = {}
for name,group in grouped_browse_2:
    browse_times_unique_ch[name] = len(group['浏览子行为编号'].unique())
    browse_times_unique_max_ch[name]=max(group['浏览子行为编号'].value_counts())
browse_times_unique_ch=DataFrame(Series(browse_times_unique_ch))
browse_times_unique_ch.reset_index(inplace=True)
browse_times_unique_ch.columns = ['用户id','不同浏览子行为次数']
#######5.获取用户不同浏览子行为最大次数#############
browse_times_unique_max_ch=DataFrame(Series(browse_times_unique_max_ch))
browse_times_unique_max_ch.reset_index(inplace=True)
browse_times_unique_max_ch.columns = ['用户id','单种子浏览最大次数']
#######6.获取用户单次最大浏览次数##################
grouped_browse_3 = browse_history_train[['用户id','时间戳']].groupby(['用户id'])
browse_times_max_single = {}
for name,group in grouped_browse_3:
    browse_times_max_single[name]=max(group['时间戳'].value_counts())
browse_times_max_single=DataFrame(Series(browse_times_max_single))
browse_times_max_single.reset_index(inplace=True)
browse_times_max_single.columns = ['用户id','单次最大浏览次数']

#######合并数据#######
data_browse = pd.merge(browse_times,browse_times_unique,on='用户id')
data_browse = pd.merge(data_browse,browse_times_unique_max,on='用户id')
data_browse = pd.merge(data_browse,browse_times_unique_ch,on='用户id')
data_browse = pd.merge(data_browse,browse_times_unique_max_ch,on='用户id')
data_browse = pd.merge(data_browse,browse_times_max_single,on='用户id')
####################browse_history_train表特征提取结束###################



#######从bank_detail_train表中获取特征#########
#######1.获取用户总交易次数####################
grouped_bank_1 = bank_detail_train.groupby('用户id')
detail_times = DataFrame(grouped_bank_1['时间戳'].count())
detail_times.reset_index(inplace=True)
detail_times.columns = ['用户id','总交易次数']
#######2.获取用户支出,收入总额,盈亏情况####################
grouped_bank_2 = bank_detail_train.groupby(['用户id','交易类型'])
detail_money   = grouped_bank_2['交易金额'].sum().unstack()
detail_money.reset_index(inplace=True)
detail_money.columns = ['用户id','收入总额','支出总额']
detail_money['盈亏'] = detail_money.apply(lambda x:1 if x['收入总额']>=x['支出总额'] else 0,axis=1)
#######3.判断收入中是否有工资收入###########
detail_wages = grouped_bank_1[['工资收入标记']].sum()
detail_wages.reset_index(inplace=True)
detail_wages.columns = ['用户id','是否有工资收入']
detail_wages['是否有工资收入'] = detail_wages['是否有工资收入'].apply(lambda x:1 if x>=1 else 0)

#######合并数据#######
data_detail = pd.merge(detail_times,detail_money,on='用户id')
data_detail = pd.merge(data_detail,detail_wages,on='用户id')
####################bbank_detail_train表特征提取结束###################



#######从bill_detail_train表中获取特征#########
#########添加两列新数据，此处的逾期与否只是上期账单与上期还款金额的比较，不代表真正的逾期###########
bill_detail_train['逾期与否'] = bill_detail_train.apply(lambda x:1 if x['上期账单金额']>x['上期还款金额'] else 0,axis=1)
bill_detail_train['循环利息'] = bill_detail_train.apply(lambda x:1 if x['循环利息']>0 else 0,axis=1)
#######1.获取银行数量#########
group_bill_1 = bill_detail_train.groupby('用户id')
bill_bank_unique = {}
for name,group in group_bill_1:
    bill_bank_unique[name] = len(group['银行id'].unique())
bill_bank_unique = DataFrame(Series(bill_bank_unique))
bill_bank_unique.reset_index(inplace=True)
bill_bank_unique.columns = ['用户id','银行数量']
#######2.获取消费笔数#########
bill_consum_times = {}
for name,group in group_bill_1:
    bill_consum_times[name] = group['消费笔数'].sum()
bill_consum_times = DataFrame(Series(bill_consum_times))
bill_consum_times.reset_index(inplace=True)
bill_consum_times.columns = ['用户id','消费笔数']
#######3.产生循环利息的银行数量#########
group_bill_2 = bill_detail_train.groupby(['用户id','银行id'])
bill_tmp_1 = group_bill_2[['循环利息']].sum()
bill_tmp_1.reset_index(inplace=True)
bill_tmp_1['循环利息'] = bill_tmp_1['循环利息'].apply(lambda x:1 if x>=1 else 0)
group_bill_3 = bill_tmp_1.groupby('用户id')
bill_bank_num_i = group_bill_3[['循环利息']].sum()
bill_bank_num_i.reset_index(inplace=True)
bill_bank_num_i.columns = ['用户id','产生循环利息银行数']
#######4.获取逾期总数量#########
bill_overdue_num = group_bill_1[['逾期与否']].sum()
bill_overdue_num.reset_index(inplace=True)
bill_overdue_num.columns = ['用户id','逾期总数量']
#######5.获取还款状态为1银行数,逾期银行数#########
bill_tmp_2 = group_bill_2[['还款状态','逾期与否']].sum()
bill_tmp_2.reset_index(inplace=True)
bill_tmp_2['还款状态'] = bill_tmp_2['还款状态'].apply(lambda x:1 if x>=1 else 0)
bill_tmp_2['逾期与否'] = bill_tmp_2['逾期与否'].apply(lambda x:1 if x>=1 else 0)
group_bill_4 = bill_tmp_2.groupby('用户id')
bill_temp_3 = group_bill_4[['还款状态']].sum()
bill_temp_4 = group_bill_4[['逾期与否']].sum()
bill_bank_num_overdue = pd.merge(bill_temp_3,bill_temp_4,left_index=True,right_index=True)
bill_bank_num_overdue.reset_index(inplace=True)
bill_bank_num_overdue.columns = ['用户id','还款状态为1银行数','逾期银行数']

#######合并数据#######
data_bill = pd.merge(bill_bank_unique,bill_consum_times,on='用户id')
data_bill = pd.merge(data_bill,bill_bank_num_i,on='用户id')
data_bill = pd.merge(data_bill,bill_overdue_num,on='用户id')
data_bill = pd.merge(data_bill,bill_bank_num_overdue,on='用户id')
####################bill_detail_train表特征提取结束###################



#######将上述所有特征数据合并#######
data = pd.merge(data_user_info,data_browse,on='用户id',how='left')
data = pd.merge(data,data_detail,on='用户id',how='left')
data = pd.merge(data,data_bill,on='用户id',how='left')
######################合并完毕，保存结果######################
data.to_csv('data_feature.csv',index=None)
###################################################