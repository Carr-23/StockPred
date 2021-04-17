
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


amd_historyAll = amd_historyAll.drop(amd_historyAll.index[tempList])
amd_historyAll = amd_historyAll.reset_index()
amd_historyAll.pop('index')
amd_historyAll.pop('Dividends')
amd_historyAll.pop('Stock Splits')

# Creating black variables
xtrainingAll = pd.DataFrame(columns=['Open', 'High', 'Low','Close', 'Volume'])
xtestingAll = pd.DataFrame(columns=['Open', 'High', 'Low','Close', 'Volume'])

ytrainingAll = pd.DataFrame(columns=['Open', 'High', 'Low','Close', 'Volume'])
ytestingAll = pd.DataFrame(columns=['Open', 'High', 'Low','Close', 'Volume'])

# Splitting Data by 50 history points
for x in range(50, amd_historyAll.shape[0]):
    if x%50 == 0:
      xtrainingAll = xtrainingAll.append(amd_historyAll.loc[x-50:x])
      xtestingAll = xtestingAll.append(amd_historyAll.iloc[x:x+1])
    else:      
      ytrainingAll = ytrainingAll.append(amd_historyAll.loc[x-50:x])
      ytestingAll = ytestingAll.append(amd_historyAll.iloc[x:x+1])

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
xtrainingAll1,xtestingAll1 = xtrainingAll.to_numpy(),xtestingAll.to_numpy()
ytrainingAll1,ytestingAll1 = ytrainingAll.to_numpy(),ytestingAll.to_numpy()

# Standarizing Data
std = StandardScaler()
xtrainingAll2 = std.fit_transform(xtrainingAll1)
xtestingAll2 = std.transform(xtestingAll1)

# Normalize, but wont change distribution and weights
scaler = MinMaxScaler(feature_range=(-1,1))
xtrainingAll3 = scaler.fit_transform(xtrainingAll2)
xtestingAll3 = scaler.transform(xtestingAll2)