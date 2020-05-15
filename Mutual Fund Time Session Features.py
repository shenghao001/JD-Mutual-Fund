# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:21:45 2020

@author: Shenghao Wang
"""
###### read fund order and click information
fund_order = pd.read_csv("dev_orders_jijin.csv")
fund_order = fund_order.astype({"fund_code": str})

fund_sku = pd.read_csv('dev_shuxing_jijin.csv')
fund_sku1 = fund_sku[['item_id', 'fund_code']]
fund_sku2 = fund_sku[['item_id', 'fund_code', 'risk_level']]

fund_sku = fund_sku.astype({"fund_code": str})

fund_click_web = pd.read_csv("C:/University of Iowa/RESEARCH/JD Financial/data/dev_jijin_web_click.csv")
fund_click_app = pd.read_table('dev_jijin_app_click.txt', delim_whitespace=True)

### Process app click data 
fund_click_app = fund_click_app.reset_index()
fund_click_app['timestamp1'] = fund_click_app['index'] + ' ' + fund_click_app['timestamp']
fund_click_app['timestamp1'] = pd.to_datetime(fund_click_app['timestamp1'])
fund_click_app = fund_click_app.drop(fund_click_app.columns[[0,1]], axis = 1)
fund_click_app = fund_click_app.rename(columns={'timestamp1': 'timestamp'})

##### timestamp process
fund_click_web['timestamp'] = pd.to_datetime(fund_click_web['timestamp'])
fund_click_app['timestamp'] = pd.to_datetime(fund_click_app['timestamp'])

## Add web and click data
fund_click = fund_click_web.append(fund_click_app)

### Order information process
fund_order1 = fund_order[['fund_code', 'succ_time', 'user_id']]
fund_sku2 = fund_sku2.drop_duplicates()
fund_order_with_risk = pd.merge(fund_order1, fund_sku2, how = 'left', on = ['fund_code'])
fund_order_with_risk = fund_order_with_risk.dropna()
fund_order_with_risk = fund_order_with_risk.rename(columns={'succ_time': 'timestamp'})
fund_order_with_risk['status'] = 1
fund_order_with_risk['timestamp'] = pd.to_datetime(fund_order_with_risk['timestamp'])
fund_click_with_risk = pd.merge(fund_click, fund_sku2, how = 'left', on = ['item_id'])
## Subset fund_click
mask = (fund_order_with_risk['timestamp'] > '2016-04-02')
fund_order_with_risk = fund_order_with_risk.loc[mask]
# append two dataset
fund_all = fund_click_with_risk.append(fund_order_with_risk)

fund_all = fund_all.sort_values('timestamp')
fund_all = fund_all.fillna('other')

### Create time session - 10 days
fund_all.sort_values(by=['user_id','timestamp'], inplace=True)
cond1 = fund_all.timestamp - fund_all.timestamp.shift(1) > pd.Timedelta(10, 'd')
cond2 = fund_all.user_id != fund_all.user_id.shift(1)
fund_all['SessionID'] = (cond1|cond2).cumsum()
fund_all = fund_all.astype({"status": str})


df1 = fund_all[fund_all['status'] == '0'].groupby(['SessionID', 'risk_level'])[['fund_code', 'item_id']].count()
df2 = fund_all[fund_all['status'] == '1'].groupby(['SessionID', 'risk_level'])[['fund_code']].count()
df3 = fund_all[fund_all['status'] == '0'].groupby(['SessionID'])[['fund_code','item_id']].nunique()
df4 = fund_all[fund_all['status'] == '0'].groupby(['SessionID', 'risk_level'])[['fund_code','item_id']].nunique()


res1 = df1.pivot_table(index=['SessionID'], columns=['risk_level'])
res2 = df2.pivot_table(index=['SessionID'], columns=['risk_level'])
res3 = df3.pivot_table(index=['SessionID'])
res4 = df4.pivot_table(index=['SessionID'], columns=['risk_level'])


flattened1 = pd.DataFrame(res1.to_records())
flattened2 = pd.DataFrame(res2.to_records())
flattened3 = pd.DataFrame(res3.to_records())
flattened4 = pd.DataFrame(res4.to_records())

final1 = flattened1.rename(columns = { "('fund_code', 1.0)":'fund_1',  "('fund_code', 2.0)":'fund_2', 
                                     "('fund_code', 3.0)":'fund_3',  "('fund_code', 4.0)":'fund_4',
                                     "('fund_code', 5.0)":'fund_5', "('fund_code', 'other')": 'fund_other',
                                     "('item_id', 1.0)":'item_1', "('item_id', 2.0)":'item_2',
                                     "('item_id', 3.0)":'item_3', "('item_id', 4.0)":'item_4',
                                     "('item_id', 5.0)":'item_5', "('item_id', 'other')":'item_other'})
final2 = flattened2.rename(columns = { "('fund_code', 1.0)":'order_1',  "('fund_code', 2.0)":'order_2', 
                                     "('fund_code', 3.0)":'order_3',  "('fund_code', 4.0)":'order_4',
                                     "('fund_code', 5.0)":'order_5', "('fund_code', 'other')":'order_other'})
final3 = flattened3

final4 = flattened4.rename(columns = { "('fund_code', 1.0)":'fund_1_unique',  "('fund_code', 2.0)":'fund_2_unique', 
                                     "('fund_code', 3.0)":'fund_3_unique',  "('fund_code', 4.0)":'fund_4_unique',
                                     "('fund_code', 5.0)":'fund_5_unique', "('fund_code', 'other')":'fund_other_unique',
                                     "('item_id', 1.0)":'item_1_unique', "('item_id', 2.0)":'item_2_unique',
                                     "('item_id', 3.0)":'item_3_unique', "('item_id', 4.0)":'item_4_unique',
                                     "('item_id', 5.0)":'item_5_unique', "('item_id', 'other')":'item_other_unique'})



final_fund = pd.concat([final1, final3, final4, final2], axis=1, sort=False)
final_fund = final_fund.fillna(0)

## user profile here is the same as I mentioned in "Product Time Session Features"
## file
fund_all = pd.merge(fund_all, user_profile, how = 'left', on = ['user_id'])
### Merge with user_data - fund
fund_all1 = pd.merge(fund_all, user_profile, how = 'left', on = ['user_id'])
final_fund_user = fund_all1.drop(fund_all1.columns[[0,1,2,3,4,5]], axis=1)
final_fund_user = final_fund_user.drop_duplicates('SessionID')
final_fund_user = final_fund_user.dropna()

final_fund = final_fund.reset_index()
## Add user feature
final_fund_with_user = pd.merge(final_fund_user, final_fund, how = 'left', on = ['SessionID'])
final_fund_with_user.to_csv("C:/University of Iowa/RESEARCH/JD Financial/DNN Training data/fund_feature_user.csv")






