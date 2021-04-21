
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
import os

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

xtrainingAll1 = xtrainingAll.copy()
xtestingAll1 = xtestingAll.copy()
ytrainingAll1 = ytrainingAll.copy()
ytestingAll1 = ytestingAll.copy()


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
xtrainingAll2,xtestingAll2 = np.array(xtrainingAll1),np.array(xtestingAll1)
ytrainingAll2,ytestingAll2 = np.array(ytrainingAll1),np.array(ytestingAll1)

# Standarizing Data
std = StandardScaler()
xtrainingAll3 = std.fit_transform(xtrainingAll2)
xtestingAll3 = std.transform(xtestingAll2)

# Normalize, but wont change distribution and weights
scaler = MinMaxScaler(feature_range=(-1,1))
xtrainingAll4 = scaler.fit_transform(xtrainingAll3)
xtestingAll4 = scaler.transform(xtestingAll3)

# Splitting Data by 50 history points with x and y division on numpy array
xtrainingAll5,xtestingAll5 = np.array(xtrainingAll4),np.array(xtestingAll4)

xtrainingAllFinal = []
xtestingAllFinal = []

ytrainingAllFinal = []
ytestingAllFinal = []

for x in range(50, xtrainingAll5.shape[0]):
    xtrainingAllFinal.append(xtrainingAll5[x-50:x])
    ytrainingAllFinal.append(ytrainingAll2[x,0])

for x in range(50, xtestingAll5.shape[0]):
    xtestingAllFinal.append(xtestingAll5[x-50:x])
    ytestingAllFinal.append(ytestingAll2[x,0])


xtrainingAllFinal,xtestingAllFinal = np.array(xtrainingAllFinal),np.array(xtestingAllFinal)
ytrainingAllFinal,ytestingAllFinal = np.array(ytrainingAllFinal),np.array(ytestingAllFinal)

if os.path.exists("model1"):
  model = tf.keras.models.load_model('model1')
else:
  # Creating Model 1
  # ? So I want to try starting the first layer with double the amount of neurons as the input size however have 50% dropout
  model = Sequential()
  # Input Layer
  model.add(LSTM(units = 50, input_shape=(xtrainingAllFinal.shape[1],xtrainingAllFinal.shape[2]), return_sequences = True))
  # Hidden Layers
  model.add(Dropout(0.2))
  model.add(LSTM(units = 70, return_sequences = True))
  model.add(Dropout(0.2))
  model.add(LSTM(units = 90, return_sequences = True))
  model.add(Dropout(0.35))
  model.add(LSTM(units = 120))
  model.add(Dropout(0.5))
  # Output Layer
  model.add(Dense(units = 1))

  model.summary()
  model.compile(optimizer = 'adam', loss = "mean_squared_error", metrics = 'mean_squared_error')
  history = model.fit(xtrainingAllFinal,ytrainingAllFinal,epochs=10,batch_size = 32, verbose = 1)

  model.save('model1')

# Test Model
yPred = model.predict(xtestingAllFinal)
print('Scaler: ',scaler.scale_)
print('MinMax: ',std)
print(scaler.scale_[0])

scl = 1/std.scale_[0]  

yPred,ytestingAllFinal = yPred*scl,ytestingAllFinal*scl

plt.figure(figsize=(14,5))
plt.plot(ytestingAllFinal, color = 'red', label = "Real Stock Price")
plt.plot(yPred, color = 'red', label = "Predicted Stock Price")
plt.title('AMD Stock Price Prediction')
plt.xlabel('Time w/ Jumps of 250')
plt.ylabel('AMD Stock Price')
plt.legend()
plt.show()
