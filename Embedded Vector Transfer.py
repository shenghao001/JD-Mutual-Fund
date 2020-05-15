# -*- coding: utf-8 -*-
"""
Created on Tue May 2 22:27:21 2020

@author: Shenghao Wang
"""

import torch  ## import pytorch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader

import csv
import sys
import itertools
import pandas
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
import time
from datetime import timedelta
from matplotlib.pyplot import scatter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib.pyplot as plt
%matplotlib inline  

'''
Step1 - Learn Reduced Product Vector
The code below is designed to learn embedded product vector by using 
Skip-gram model
'''


## read click_chain file for three products
df1 = pd.read_csv('shampoo_click_chain.csv')
df2 = pd.read_csv('washer_click_chain.csv')
df3 = pd.read_csv('toothpaste_click_chain.csv')

## change products type
df1['brand_code'] = df1['brand_code'].astype(str)
df2['brand_code'] = df2['brand_code'].astype(str)
df3['brand_code'] = df3['brand_code'].astype(str)

### get products and products indices
vocabulary = df3['brand_code'].drop_duplicates().tolist()
word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
vocabulary_size = len(vocabulary)

### sliding the windows by time - t = 30 mins
### For each click, pair all clicks within 30 minutes
import datetime as dt

def window(x): # x is the time duration 
    for i in range(len(new_click1.index)):
        W1 = new_click1.iloc[i,1] - dt.timedelta(minutes = x)
        W2 = new_click1.iloc[i,1] + dt.timedelta(minutes = x)
        df_sub = new_click1[(new_click1['request_time'] >= W1) & (new_click1['request_time'] <= W2)]
        idx_pairs = []
        window_size = len(df_sub.index)-1
        indices = [word2idx[product] for product in df_sub['sku_ID']]
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out products
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))
    idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array
    return idx_pairs
        
        
### one-hot encoding generator
def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

## training process for getting reduced vector for each product
embedding_dims = 2 # the dimension may vary
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 50
learning_rate = 0.01

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:    
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')

        
'''
Step2 - Learn user embedded vector. We use the learned product vector to 
train the embedded user embedded vector
'''
## Build model

class embeddingModel(nn.Module):

    def __init__(self, embedding_dim, user_size, feature_size):
        super(embeddingModel, self).__init__()
        self.embeddings = nn.Embedding(user_size, embedding_dim)
        self.linear_s = nn.Linear(embedding_dim, feature_size)
        self.linear_t = nn.Linear(embedding_dim, feature_size)
        self.linear_w = nn.Linear(embedding_dim, feature_size)


    def forward(self, inputs1， inputs2):
        embeds = self.embeddings(inputs1)
        if inputs2 =='xfs':
            out = self.linear_s(embeds)
        if inputs2 == 'yg':
            out = self.linear_t(embeds)
        if inputs2 =='xyj':
            out = self.linear_w(embeds)
        return out


## implement the model    
    
USER_SIZE = 9783 # This number may vary
EMBEDDING_DIM = 2 # This number may vary
FEATURE_SIZE = 2 # This number may vary
    
losses = []
loss_function = torch.nn.MSELoss()
model = embeddingModel(EMBEDDING_DIM, USER_SIZE, FEATURE_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
#     counter = 0
#     counter1 = 0
    for i in range(len(f3.index)):
        user_id = torch.tensor(f3['user'][i], dtype=torch.long)
        feature_id = torch.tensor(f3['feature'][i], dtype=torch.long)
        type1 = f3['type'][i]
        
        model.zero_grad() 

        # Step 3. Run the forward pass, getting output
        out = model(user_id， type1)

        # Step 4. Compute your loss function.
        loss = loss_function(out, feature_id.type(torch.FloatTensor))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(loss.item())

    losses.append(total_loss)
print(losses)  
# The loss decreased every iteration over the training data!
































