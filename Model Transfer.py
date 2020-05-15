# -*- coding: utf-8 -*-
"""
Created on Tue May 2 10:50:02 2020

@author: Shenghao Wang
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils import np_utils
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler 
from keras.optimizers import Adam, SGD
from numpy.random import seed
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import r2_score
from keras.layers import LSTM
from keras import backend as K
from keras.layers import Concatenate
from keras.models import Model, load_model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.optimizers import Adagrad
from keras.utils.conv_utils import convert_kernel
from tensorflow.keras import layers
from tensorflow import keras
from keras.utils import plot_model
import keras.layers
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.layers import Activation, Dense
from keras import backend as K
from numpy.random import seed
from sklearn.model_selection import train_test_split as train_test_split
from sklearn.preprocessing import MinMaxScaler


'''
Model Transfer Part 1
The code below is designed for training shampoo product features to predict
shampoo purchase situation. 

We train the model and save the layers except for output layer. And we will
reuse the layers for transfer learning
'''


#### define data for shampoo / toothpaste / washer
# In this case, we use shampoo as example

def M_XFS():
    xyj_final = pd.read_csv("C:/University of Iowa/PhD Research Projects/JD Mutual Fund Prediction/JD Financial/Knowledge Transfer Data/M_XFS.csv")
    dataset = xyj_final.drop(xyj_final.columns[[0,1,2,6,7,8,9,10]],axis = 1)
    df1 = dataset.pop('#order') # remove column b and store it in df1
    dataset['#order']=df1 # add b series as a 'new' column.
    dataset.columns
    # creating input features and target variables
    X= dataset.iloc[:,0:21]
    y= dataset.iloc[:,21]

    #standardizing the input feature 
    sc = StandardScaler()
    X = sc.fit_transform(X)

    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=1)
    
    X_train, X_val, y_train, y_val  = train_test_split(trainX, trainY, test_size=0.2, random_state=1)
    
 
    return X_train, y_train, X_val, y_val


### Define Training process for Shampoo
def model_M_XFS(X_train, y_train, X_val, y_val):
    
    seed(10000)
    
    model2 = Sequential()
    model2.add(Dense({{choice([25,30,35,40,42])}}, input_shape=(21,)))
    model2.add(Activation({{choice(['relu'])}}))
    model2.add(Dropout({{uniform(0, 1)}}))
    model2.add(Dense({{choice([20,25,30,35,40,])}}))
    model2.add(Activation({{choice(['relu'])}}))
    model2.add(Dropout({{uniform(0, 1)}}))
    
    if ({{choice(['two', 'three'])}}) == 'three':
        model2.add(Dense({{choice([15,20,25,30,35])}}))
        model2.add(Activation({{choice(['relu'])}}))
        model2.add(Dropout({{uniform(0, 1)}}))
        
    model2.add(Dense(1))
    model2.add(Activation('relu'))
    
    adam = keras.optimizers.Adam(lr={{choice([0.01, 0.001, 0.0001])}})
    rmsprop = keras.optimizers.RMSprop(lr={{choice([0.01, 0.001, 0.0001])}})
    sgd = keras.optimizers.SGD(lr={{choice([0.01, 0.001, 0.0001])}})
   
    choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'rmsprop':
        optim = rmsprop
    else:
        optim = sgd

    model2.compile(loss='mse', metrics=['mse','mae'], optimizer=optim)
    result = model2.fit(X_train, y_train,
                      batch_size={{choice([128, 256, 512])}},
                      nb_epoch = 100,
                      verbose=2,
                      validation_data = (X_val, y_val))
    loss, mse, mae = model2.evaluate(X_val, y_val, verbose=0)
    
    validation_mse = np.amin(result.history['val_mean_squared_error']) 
    validation_mae = np.amin(result.history['val_mean_squared_error'])   
    print('Best validation mse of epoch:', validation_mse)
    print('Best validation mse of epoch:', validation_mae)
    
    return {'loss': loss, 'status': STATUS_OK, 'model': model2}



### fit the model and produce output
X_train, y_train, X_val, y_val = M_XFS()

## Excecution
best_run_MXFS, best_model_MXFS= optim.minimize(model = model_M_XFS,
                                                     data = M_XFS,
                                                     algo = tpe.suggest,
                                                     max_evals = 30,
                                                     trials = Trials(),
                                                     notebook_name = 'Transfer Learning Model')



'''
Model Transfer Part 2 - save previous layer and only change output layer
The time, the target variable becomes purchase decision of mutual fund

'''
trained_shampoo = load_model('shampoo_trained.hdf5')
shampoo_weight1 = best_model_MXFS.layers[0].get_weights()
shampoo_weight2 = best_model_MXFS.layers[3].get_weights()


# ##### Shampoo predict Shampoo Target Variable continuous
M = pd.read_csv("C:/University of Iowa/PhD Research Projects/JD Mutual Fund Prediction/JD Financial/Knowledge Transfer Data/OnlyM.csv")
dataset = M.drop(M.columns[[0,1,2,6,7,8,9,10]],axis = 1)

# creating input features and target variables
X= dataset.iloc[:,0:7]
y= dataset.iloc[:,7]

#standardizing the input feature 
sc = StandardScaler()
X = sc.fit_transform(X)

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val  = train_test_split(trainX, trainY, test_size=0.2, random_state=1)


###Load Previous shampoo Weights
trained_shampoo = load_model('shampoo_trained.hdf5')
shampoo_weight1 = trained_shampoo.layers[0].get_weights()
shampoo_weight2 = trained_shampoo.layers[3].get_weights()

input1 = Input(shape=(7, ))

#set weights for first layer
first_shampoo = Dense(13, activation = 'relu')
out_first_shampoo = first_shampoo(input1)
first_shampoo.set_weights(shampoo_weight1)


# set weights for second layer
second_shampoo = Dense(10, activation = 'relu')
out_second_shampoo = second_shampoo(out_first_shampoo)
second_shampoo.set_weights(shampoo_weight2)

#add third layer

third_shampoo = Dense(10, activation = 'relu')
#third_shampoo = Dense(10, activation = 'relu')
out_third_shampoo = third_shampoo(out_second_shampoo)
out_third_shampoo = Dropout(0.1)(out_third_shampoo)

fourth_shampoo = Dense(6, activation = 'relu')
#third_shampoo = Dense(10, activation = 'relu')
out_fourth_shampoo = fourth_shampoo(out_third_shampoo)
out_fourth_shampoo = Dropout(0.1)(out_fourth_shampoo)


#output layer        
fifth_shampoo = Dense(1, activation = 'relu')
out_fifth_shampoo = fifth_shampoo(out_fourth_shampoo)

#Merge and run

merged_model = Model([input1], outputs = out_fifth_shampoo)
merged_model.compile(loss='mse', metrics=['mse'], optimizer = Adam(lr=0.01))

# checkpoint
filepath = "shampoo_update.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor= 'val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

merged_model.fit(X_train, y_train,
                 batch_size = 128,
                 nb_epoch = 150,
                 verbose = 2,
                 validation_data = (X_val, y_val),
                 callbacks = callbacks_list)


















