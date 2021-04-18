
# !pip install yahoofinancials
# !pip install yfinance
# !pip install --upgrade tensorflow

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Get Data
amd_stock = yf.Ticker('AMD')
amd_historyAll = amd_stock.history(period="3y").reset_index()

# Delete date since numbers are just fine
amd_historyAll.pop('Date')

# Round Data to be divisible by 50
rounded = 50 * round(len(amd_historyAll)/50)
limitOfDelete = len(amd_historyAll) - rounded
tempList = list(range(0,limitOfDelete))

# Drop Date (useless information that the model doesn't need to know)
amd_historyAll = amd_historyAll.drop(amd_historyAll.index[tempList])
amd_historyAll = amd_historyAll.reset_index()
amd_historyAll.pop('index')
amd_historyAll.pop('Dividends')
amd_historyAll.pop('Stock Splits')

# Creating blank variables
xtrainingAll = pd.DataFrame(columns=['Open', 'High', 'Low','Close', 'Volume'])
xtestingAll = pd.DataFrame(columns=['Open', 'High', 'Low','Close', 'Volume'])

ytrainingAll = pd.DataFrame(columns=['Open', 'High', 'Low','Close', 'Volume'])
ytestingAll = pd.DataFrame(columns=['Open', 'High', 'Low','Close', 'Volume'])

multiplier = 0
# Splitting Data by 50 history points
for x in range(amd_historyAll.shape[0]):
    if x >= multiplier * 250 and x < multiplier * 250 + 50:
      xtestingAll = xtestingAll.append(amd_historyAll.iloc[x])
      ytestingAll = xtestingAll.append(amd_historyAll.iloc[x])
    else:      
      xtrainingAll = xtrainingAll.append(amd_historyAll.loc[x])
      ytrainingAll = ytrainingAll.append(amd_historyAll.loc[x])

    if x == 250 * multiplier + 49:
      multiplier+=1

xHistory = [xtrainingAll,xtestingAll]
yHistory = [ytrainingAll,ytestingAll]

# Deleting the useless columns
for x in xHistory:
  x.pop('Close')

for y in yHistory:
  y.pop('Open')
  y.pop('High')
  y.pop('Low')
  y.pop('Volume')

# Convert DataFrame to numpy array
xtrainingAll2,xtestingAll2 = np.array(xtrainingAll),np.array(xtestingAll)
ytrainingAll2,ytestingAll2 = np.array(ytrainingAll),np.array(ytestingAll)

# Standarizing Data
std = StandardScaler()
xtrainingAll3 = std.fit_transform(xtrainingAll2)
xtestingAll3 = std.transform(xtestingAll2)

# Normalize, but wont change distribution and weights
scaler = MinMaxScaler(feature_range=(-1,1))
xtrainingAll4 = scaler.fit_transform(xtrainingAll3)
xtestingAll4 = scaler.transform(xtestingAll3)

# Splitting Data by 50 history points with x and y division on numpy array
xtrainingAll5,xtestingAll5 = np.array(xtrainingAll5),np.array(xtestingAll4)

xtrainingAllFinal = []
xtestingAllFinal = []

ytrainingAllFinal = []
ytestingAllFinal = []

for x in range(50, xtrainingAll5.shape[0]):
    xtrainingAllFinal.append(xtrainingAll5[x-50:x])
    ytrainingAllFinal.append(ytrainingAll2[x,0])

for x in range(50, xtestingAll1.shape[0]):
    xtestingAllFinal.append(xtestingAll5[x-50:x])
    ytestingAllFinal.append(ytestingAll2[x,0])


xtrainingAllFinal,xtestingAllFinal = np.array(xtrainingAllFinal),np.array(xtestingAllFinal)
ytrainingAllFinal,ytestingAllFinal = np.array(ytrainingAllFinal),np.array(ytestingAllFinal)

# Creating Model
# ? So I want to try starting the first layer with double the amount of neurons as the input size however have 50% dropout

model = Sequential()

# Input Layer

model.add(LSTM(xtrainingAll3.shape[0] * 2, input_shape=(xtrainingAll3.shape[])))

# TODO: 
# Divide the data as a dataframe
# Do the normalization and standardization on the new split data frame
# Do the list division on the numpy array
# We should be left with our 3D data 