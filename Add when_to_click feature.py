# -*- coding: utf-8 -*-
"""
Created on Thu May 10 19:08:07 2020

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
The code below is designed for extracting user shopping behavior features. 
The features we extract are related with shampoo. You may use the code to
extract features for washer or toothpaste. 

'''


## Read file
xfs_click_app = pandas.read_table('dev_app_xifashui.txt', delim_whitespace=True)
xyj_click_app = pandas.read_table('dev_app_xiyiji.txt', delim_whitespace=True)
yg_click_app = pandas.read_table('dev_app_yagao.txt', delim_whitespace=True)
xfs_click_web = pandas.read_table('dev_web_xifashui.txt', delim_whitespace=True)
xyj_click_web = pandas.read_table('dev_web_xiyiji.txt', delim_whitespace=True)
yg_click_web = pandas.read_table('dev_web_yagao.txt', delim_whitespace=True)


fund_click_web = pandas.read_table('dev_jijin_web_click.txt', delim_whitespace=True)
fund_click_app = pandas.read_table('dev_jijin_app_click.txt', delim_whitespace=True)


xfs_order = pd.read_csv("dev_orders_xifashui.csv")
xyj_order = pd.read_csv("dev_orders_xiyiji.csv")
yg_order = pd.read_csv("dev_orders_yagao.csv")

xfs_order = xfs_order.drop(columns = ['sku_name'])
xyj_order = xyj_order.drop(columns = ['sku_name'])
yg_order = yg_order.drop(columns = ['sku_name'])

fund_order = pd.read_csv("dev_orders_jijin.csv")
fund_sku = pd.read_csv('dev_shuxing_jijin.csv')
fund_sku1 = fund_sku[['item_id', 'fund_code']]
fund_sku2 = fund_sku[['item_id', 'fund_code', 'risk_level']]

##### Read app and web clickstream data
xfs_click_app = pd.read_csv("C:/University of Iowa/PhD Research Projects/JD Mutual Fund Prediction/JD Financial/data/dev_app_xifashui.csv")
xfs_click_web = pandas.read_table('dev_web_xifashui.txt', delim_whitespace=True)

# Convert string to datetime, no need to pass any arguments 
xfs_click_app['timestamp'] = pd.to_datetime(xfs_click_app['timestamp'])

# convert unix epoch time to datetime
xfs_click_web['timestamp'] = pd.to_datetime(xfs_click_web['timestamp'],unit='ms')

## Compute Total Unique Item and brand visited for XFS
#Step 1
xfs = pd.read_csv("C:/University of Iowa/PhD Research Projects/JD Mutual Fund Prediction/JD Financial/data/dev_sku_xifashui.csv")
xfs = xfs[['sku_id','sku_name','brand_code']]
xfs_click_web = pd.merge(xfs_click_web, xfs, how = 'left', on = ['sku_id'])
xfs_click_app = pd.merge(xfs_click_app, xfs, how = 'left', on = ['sku_id'])
xfs_click_web = xfs_click_web.dropna()
xfs_click_web1 = xfs_click_web[['user_id','sku_id','brand_code']]
xfs_click_app = xfs_click_app.dropna()
xfs_click_app1 = xfs_click_app[['user_id','sku_id','brand_code']]


## Step 2
df1 = xfs_click_web1.groupby('user_id').agg(['nunique']).stack()
df2 = xfs_click_app1.groupby('user_id').agg(['nunique']).stack()
df_xfs_click = pd.merge(df1, df2, how='left', on=['user_id'])
df_xfs_click = df_xfs_click.fillna(0)
df_xfs_click['unique_item'] = df_xfs_click['sku_id_x'] + df_xfs_click['sku_id_y']
df_xfs_click['unique_brand'] = df_xfs_click['brand_code_x'] + df_xfs_click['brand_code_y']
df_xfs_click = df_xfs_click.drop(['sku_id_x', 'sku_id_y', 'brand_code_x', 'brand_code_y'], axis = 1)
df_xfs_click1 = df_xfs_click.reset_index()
xfs_time = xfs_click_web.append(xfs_click_app)


### add shampoo's user when to click features 
xfs_time.timestamp=pd.to_datetime(xfs_time.timestamp)
df2 = xfs_time.groupby([pd.Grouper(key='timestamp',freq='6H'),xfs_time.user_id]).size().reset_index(name='count')
df2['hour'] = df2['timestamp'].dt.hour
df2 = df2[['user_id', 'count', 'hour']]
df_shampoo = df2.groupby(["user_id",'hour'])['count'].sum()
df_shampoo1 = df_shampoo.to_frame()
res = df_shampoo1.pivot_table(index=['user_id'], columns=['hour'])
flattened = pd.DataFrame(res.to_records())

final2 = flattened.rename(columns = { "('count', 0)":'0-6',  "('count', 6)":'6-12', "('count', 12)":'12-18',  
                                     "('count', 18)":'18-24'})
final2 = final2.fillna(0)
final2[['0-6', '6-12', '12-18', '18-24']] = final2[['0-6', '6-12', '12-18', '18-24']].apply(lambda x: x/x.sum(), axis=1)

## final2 is the dataframe containing the 'when to click' feature
final2





