# -*- coding: utf-8 -*-
"""
Created on Thu May 10 23:40:25 2020

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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

#from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
#from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

'''
The code below is designed for extracting the features by rolling window. 
Each click will incur a new prediction.

We use mutual fund as example. You may use this code to extract the other
products features, such as shampoo, washer, and toothpaste. 
'''
fund_web = pd.read_csv("C:/University of Iowa/PhD Research Projects/JD Mutual Fund Prediction/JD Financial/data/dev_jijin_web_click.csv")
fund_click_app = pd.read_table('dev_jijin_app_click.txt', delim_whitespace=True)
fund_click_app = fund_click_app.reset_index()
fund_click_app['timestamp1'] = fund_click_app['index'] + ' ' + fund_click_app['timestamp']
fund_click_app['timestamp1'] = pd.to_datetime(fund_click_app['timestamp1'])
fund_click_app = fund_click_app.drop(fund_click_app.columns[[0,1]], axis = 1)
fund_click_app = fund_click_app.rename(columns={'timestamp1': 'timestamp'})
fund_click_app = fund_click_app.drop_duplicates()

fund_click_app['source'] = 'app'
fund_web['source'] = 'web'
fund_click = fund_click_app.append(fund_web)
fund_click['timestamp'] = pd.to_datetime(fund_click['timestamp'])
fund_click['status'] = 0


fund_order = pd.read_csv("dev_orders_jijin.csv")
fund_sku = pd.read_csv('dev_shuxing_jijin.csv')
fund_sku1 = fund_sku[['item_id', 'fund_code']]
fund_sku2 = fund_sku[['item_id', 'fund_code', 'risk_level']]
fund_sku3 = fund_sku[['item_id', 'risk_level']]

fund_order['fund_code'] = fund_order['fund_code'].astype(str)
fund_sku2['fund_code'] = fund_sku2['fund_code'].astype(str)

fund_sku2_new = fund_sku2.drop_duplicates()
fund_order1 = pd.merge(fund_order, fund_sku2_new, on = ['fund_code'], how = 'left')
fund_order_update = fund_order1[['item_id', 'succ_time', 'user_id', 'risk_level']]
fund_order_update['status'] = 1

fund_order_update1 = fund_order_update.drop_duplicates()
fund_order_update1 = fund_order_update1.dropna()
fund_click_new = pd.merge(fund_click, fund_sku2_new, on = ['item_id'], how = 'left')
fund_click_new1 = fund_click_new[['item_id', 'timestamp','source','user_id','risk_level', 'status']]
fund_click_new1 = fund_click_new1.dropna()
fund_order_update1 = fund_order_update1.rename(columns={'succ_time': 'timestamp'})

# fund click and buy data generate and save
# select time after 2016-05-01 for order data
fund_order_update2 = fund_order_update1[fund_order_update1['timestamp'] >= '2016-05-01']
fund = fund_order_update2.append(fund_click_new1)
fund['timestamp'] = pd.to_datetime(fund['timestamp'])
fund = fund.sort_values('timestamp')
fund1 = fund.reset_index(drop=True)



fund1['hour'] = fund1['timestamp'].dt.hour
# Add click hour range
fund1['hour0'] = [1 if x >= 0 and x < 6 else 0 for x in fund1['hour']]
fund1['hour6'] = [1 if x >= 6 and x < 12 else 0 for x in fund1['hour']]
fund1['hour12'] = [1 if x >= 12 and x < 18 else 0 for x in fund1['hour']]
fund1['hour18'] = [1 if x >= 18 and x < 24 else 0 for x in fund1['hour']]
# Add risk level count
fund1['risk1'] = [1 if x == 1 else 0 for x in fund1['risk_level']]
fund1['risk2'] = [1 if x == 2 else 0 for x in fund1['risk_level']]
fund1['risk3'] = [1 if x == 3 else 0 for x in fund1['risk_level']]
fund1['risk4'] = [1 if x == 4 else 0 for x in fund1['risk_level']]
fund1['risk5'] = [1 if x == 5 else 0 for x in fund1['risk_level']]
# Add source count
fund1['app'] = [1 if x == 'app' else 0 for x in fund1['source']]
fund1['web'] = [1 if x == 'web' else 0 for x in fund1['source']]

## Compute Total Unique item / app click / web click / risk level distribution

## Backward rolling with windows = 3 for app click / web click and #unique fund
fund_click_item = fund1.set_index('timestamp').groupby('user_id').item_id.rolling("7D").apply(lambda x: len(np.unique(x)))
fund_clickapp = fund1.set_index('timestamp').groupby('user_id').app.rolling("7D").apply(lambda x: (x > 0).sum())
fund_clickweb = fund1.set_index('timestamp').groupby('user_id').web.rolling("7D").apply(lambda x: (x > 0).sum())
## Backward rolling with windows = 3 for risk and hour
fund_risk1 = fund1.set_index('timestamp').groupby('user_id').risk1.rolling("7D").apply(lambda x: (x > 0).sum())
fund_risk2 = fund1.set_index('timestamp').groupby('user_id').risk2.rolling("7D").apply(lambda x: (x > 0).sum())
fund_risk3 = fund1.set_index('timestamp').groupby('user_id').risk3.rolling("7D").apply(lambda x: (x > 0).sum())
fund_risk4 = fund1.set_index('timestamp').groupby('user_id').risk4.rolling("7D").apply(lambda x: (x > 0).sum())
fund_risk5 = fund1.set_index('timestamp').groupby('user_id').risk5.rolling("7D").apply(lambda x: (x > 0).sum())

fund_hour0 = fund1.set_index('timestamp').groupby('user_id').hour0.rolling("7D").apply(lambda x: (x > 0).sum())
fund_hour6 = fund1.set_index('timestamp').groupby('user_id').hour6.rolling("7D").apply(lambda x: (x > 0).sum())
fund_hour12 = fund1.set_index('timestamp').groupby('user_id').hour12.rolling("7D").apply(lambda x: (x > 0).sum())
fund_hour18 = fund1.set_index('timestamp').groupby('user_id').hour18.rolling("7D").apply(lambda x: (x > 0).sum())

# Calculate if there is buy in next 24 hours
fund_new = fund1.set_index('timestamp')
df2 = fund_new[::-1]
df2.index = pd.datetime(2050,1,1) - df2.index
df2 = df2.groupby('user_id').rolling('24H', min_periods=0).status.sum()
df3 = df2[::-1]
df3.index = fund_new.index


# convert to dataframe
#fund_buy1 = df3.to_frame().sort_values(['user_id','timestamp']).reset_index(drop = True)
fund_click_item1 = fund_click_item.to_frame().sort_values(['user_id','timestamp']).reset_index()
fund_clickapp1 = fund_clickapp.to_frame().sort_values(['user_id','timestamp']).reset_index(drop = True)
fund_clickweb1 = fund_clickweb.to_frame().sort_values(['user_id','timestamp']).reset_index(drop = True)

fund_risk_level1 = fund_risk1.to_frame().sort_values(['user_id','timestamp']).reset_index(drop = True)
fund_risk_level2 = fund_risk2.to_frame().sort_values(['user_id','timestamp']).reset_index(drop = True)
fund_risk_level3 = fund_risk3.to_frame().sort_values(['user_id','timestamp']).reset_index(drop = True)
fund_risk_level4 = fund_risk4.to_frame().sort_values(['user_id','timestamp']).reset_index(drop = True)
fund_risk_level5 = fund_risk5.to_frame().sort_values(['user_id','timestamp']).reset_index(drop = True)

fund_hour_level0 = fund_hour0.to_frame().sort_values(['user_id','timestamp']).reset_index(drop = True)
fund_hour_level6 = fund_hour6.to_frame().sort_values(['user_id','timestamp']).reset_index(drop = True)
fund_hour_level12 = fund_hour12.to_frame().sort_values(['user_id','timestamp']).reset_index(drop = True)
fund_hour_level18 = fund_hour18.to_frame().sort_values(['user_id','timestamp']).reset_index(drop = True)

index = fund1.sort_values(['user_id','timestamp'])['status'].to_frame().reset_index(drop=True)
index = index.rename(columns={'status': 'ifbuy'})

df5 = df3.drop(df3.columns[[0,1]],axis=1)
combine = [fund_click_item1, fund_clickapp1, fund_clickweb1, fund_risk_level1, fund_risk_level2, fund_risk_level3, 
           fund_risk_level4, fund_risk_level5, fund_hour_level0, fund_hour_level6, fund_hour_level12, 
           fund_hour_level18, df5, index]

merged_df = pd.concat(combine, axis=1)
fund_update = merged_df[merged_df['ifbuy'] == 0]
fund_update1 = fund_update.drop(fund_update.columns[-1],axis=1)

fund_final = fund_update1.rename(columns={'status': '#order'})










