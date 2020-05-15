# -*- coding: utf-8 -*-
"""
Created on Tue May 2 21:45:04 2020

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
The Code below is the OLS regression model, depends on different product input
The example below is about using mutual fund features to predict purchase
of mutual fund. You may use this code for predicting other target variables.
'''

import pandas as pd
import statsmodels.formula.api as sm
final = pd.read_csv("C:/University of Iowa/PhD Research Projects/JD Mutual Fund Prediction/JD Financial/Knowledge Transfer Data/onlyM.csv")
X = final[['item_id', 'app','web', 'risk1', 'risk2', 'risk3', 'risk4', 'risk5', 'hour0', 'hour6', 'hour12', 'hour18']]
Y = np.array(final[['#order']])
#X = sm.add_constant(X)
est = sm.OLS(Y, X).fit()
est.summary()






























