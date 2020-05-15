# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:18:32 2020

@author: Shenghao Wang
"""

import csv
import sys
import itertools
import pandas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
import time
from datetime import timedelta

'''
The code below is designed for creating the time session based dataframe, The processed 
product is washer. You may do the same step for shampoo and toothapste. 
'''


### Read File
### Read app and web clickstream data
xyj_click_app = pd.read_csv("C:/University of Iowa/RESEARCH/JD Financial/data/dev_app_xiyiji.csv")
xyj_click_web = pandas.read_table('dev_web_xiyiji.txt', delim_whitespace=True)

# Convert string to datetime, no need to pass any arguments 
xyj_click_app['timestamp'] = pd.to_datetime(xyj_click_app['timestamp'])

# convert unix epoch time to datetime
xyj_click_web['timestamp'] = pd.to_datetime(xyj_click_web['timestamp'],unit='ms')

# read order information
xyj_order = pd.read_csv("dev_orders_xiyiji.csv")


'''
Read the user profile. User features has already been converted to one-hot
encoding. 
'''
user_profile = pd.read_csv("C:/University of Iowa/RESEARCH/JD Financial/user_profile.csv")

# Washer information file read
xyj_info = pd.merge(xyj_order, xyj, how = 'left', on = ['sku_name'] )
xyj_info = xyj_info[['user_id','timestamp', 'sku_name','price_rate', 'sku_id', 'brand_code']]
xyj_info = xyj_info.dropna()
xyj_info1 = xyj_info[['user_id','timestamp','price_rate', 'sku_id', 'brand_code']]
xyj_info1['status'] = '1'
xyj_info1 = xyj_info1.sort_values('timestamp')
xyj_info1 = xyj_info1.reset_index(drop = True)


###By click gap
### Merge washer click data with price rate

## use merger_asof pandas to find nearest timestamp
xyj_click = xyj_click_web.append(xyj_click_app)
xyj_click['status'] = '0'
xyj_click = xyj_click.sort_values('timestamp')
xyj_click['sku_id'] = xyj_click['sku_id'].astype(int)
xyj_info1['sku_id'] = xyj_info1['sku_id'].astype(int)
washer_click = pd.merge_asof(xyj_click, xyj_info1, on = ['timestamp'], by = 'sku_id', direction='nearest')
washer_click = washer_click[['timestamp', 'sku_id', 'brand_code', 'user_id_x', 'status_x', 'price_rate']]
washer_click = washer_click.rename({'user_id_x':'user_id', 'status_x':'status'}, axis=1)
washer_session_all = washer_click.append(xfs_info1)
washer_session_all = washer_session_all.sort_values('timestamp')
washer_session_all = washer_session_all.fillna('other')


# Create time session for washer - 7 days
washer_session_all.sort_values(by=['user_id','timestamp'], inplace=True)
cond1 = washer_session_all.timestamp - washer_session_all.timestamp.shift(1) > pd.Timedelta(7, 'd')
cond2 = washer_session_all.user_id != washer_session_all.user_id.shift(1)
washer_session_all['SessionID'] = (cond1|cond2).cumsum()


df1 = washer_session_all[washer_session_all['status'] == '0'].groupby(['SessionID', 'price_rate'])[['sku_id', 'brand_code']].count()
df2 = washer_session_all[washer_session_all['status'] == '1'].groupby(['SessionID', 'price_rate'])[['sku_id']].count()
df3 = washer_session_all[washer_session_all['status'] == '0'].groupby(['SessionID'])[['sku_id','brand_code']].nunique()
df4 = washer_session_all[washer_session_all['status'] == '0'].groupby(['SessionID', 'price_rate'])[['sku_id','brand_code']].nunique()

res1 = df1.pivot_table(index=['SessionID'], columns=['price_rate'])
res2 = df2.pivot_table(index=['SessionID'], columns=['price_rate'])
res3 = df3.pivot_table(index=['SessionID'])
res4 = df4.pivot_table(index=['SessionID'], columns=['price_rate'])


flattened1 = pd.DataFrame(res1.to_records())
flattened2 = pd.DataFrame(res2.to_records())
flattened3 = pd.DataFrame(res3.to_records())
flattened4 = pd.DataFrame(res4.to_records())

### rename and dataframe

final1 = flattened1.rename(columns = { "('sku_id', 1.0)":'item_1',  "('sku_id', 2.0)":'item_2', 
                                     "('sku_id', 3.0)":'item_3',  "('sku_id', 4.0)":'item_4',
                                     "('sku_id', 5.0)":'item_5', "('sku_id', 'other')":'item_other',
                                     "('brand_code', 1.0)":'brand_1', "('brand_code', 2.0)":'brand_2',
                                     "('brand_code', 3.0)":'brand_3', "('brand_code', 4.0)":'brand_4',
                                     "('brand_code', 5.0)":'brand_5', "('brand_code', 'other')":'brand_other'})
final2 = flattened2.rename(columns = { "('sku_id', 1.0)":'order_1',  "('sku_id', 2.0)":'order_2', 
                                     "('sku_id', 3.0)":'order_3',  "('sku_id', 4.0)":'order_4',
                                     "('sku_id', 5.0)":'order_5', "('sku_id', 'other')":'order_other'})
final3 = flattened3

final4 = flattened4.rename(columns = { "('sku_id', 1.0)":'item_1_unique',  "('sku_id', 2.0)":'item_2_unique', 
                                     "('sku_id', 3.0)":'item_3_unique',  "('sku_id', 4.0)":'item_4_unique',
                                     "('sku_id', 5.0)":'item_5_unique', "('sku_id', 'other')":'item_other_unique',
                                     "('brand_code', 1.0)":'brand_1_unique', "('brand_code', 2.0)":'brand_2_unique',
                                     "('brand_code', 3.0)":'brand_3_unique', "('brand_code', 4.0)":'brand_4_unique',
                                     "('brand_code', 5.0)":'brand_5_unique', "('brand_code', 'other')":'brand_other_unique'})


### Merge with user_data - Washer
washer_session_all1 = pd.merge(washer_session_all, user_profile, how = 'left', on = ['user_id'])
washer_session_with_user = washer_session_all1.drop(washer_session_all1.columns[[0,1,2,3,4,5]], axis=1)
washer_session_with_user = washer_session_with_user.drop_duplicates('SessionID')
washer_session_with_user = washer_session_with_user.dropna()

### concatenate 4 dataframes

final_washer = pd.concat([final1, final3, final4, final2], axis=1, sort=False)
final_washer = final_washer.fillna(0)
final_washer = final_washer.reset_index()

## Add user feature and save the file to CSV
final_washer_with_user = pd.merge(washer_session_with_user, final_washer, how = 'left', on = ['SessionID'])

final_washer_with_user.to_csv("C:/University of Iowa/RESEARCH/JD Financial/DNN Training data/washer_feature_user.csv")





