
# !pip install yahoofinancials
# !pip install yfinance
# !pip install --upgrade tensorflow

print('test')
import numpy as np
import pandas as pd
#from yahoofinancials import YahooFinancial
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

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

# Splitting Data by 50 history points
trainingAll = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'])
validationAll = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'])
testingAll = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'])

for x in range(int(rounded/50)):
    trainingAll = trainingAll.append(amd_historyAll.loc[x*50:x*50+35])
    validationAll = validationAll.append(amd_historyAll.iloc[x*50+35:x*50+41])
    testingAll = testingAll.append(amd_historyAll.iloc[x*50+40:x*50+50])
    

#trainingAll = amd_historyAll[:int(len(amd_historyAll)*0.7)]
#validationAll = amd_historyAll[int(len(amd_historyAll)*0.7):-int(len(amd_historyAll)*0.2)] 
#testingAll = amd_historyAll[-int(len(amd_historyAll)*0.2):] 

# Normalize Data
scaler = MinMaxScaler()
trainingAll = scaler.fit_transform(trainingAll)
validationAll = scaler.transform(validationAll)
testingAll = scaler.transform(testingAll)

# Print Statements
testingAll,validationAll,testingAll = np.array(testingAll),np.array(validationAll),np.array(testingAll)
print("--------------------Train------------------------")
print(trainingAll)
print("--------------------Val------------------------")
print(validationAll)
print("--------------------Test------------------------")
print(testingAll)

print("--------------------OG------------------------")
print(amd_historyAll)
print("--------------------OG------------------------")
print(amd_historyAll.iloc[0:4])