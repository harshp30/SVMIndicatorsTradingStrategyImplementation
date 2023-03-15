# SVM with SMA and RSI Trading Strategy

import numpy as np
import pandas as pd
import datetime
import yfinance as yf
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

'''
Function to get a specific stock based on started and end date
This returns a data frame containing all information for the specific chosen stock
'''
def fetch_data(symbol, start, end):
    ticker = yf.download(symbol, start, end)
    return pd.DataFrame(ticker)

'''
Function to calculate the rsi (relative strength index) for the fetched stock 
'''
def calculate_rsi(data, period=14):
    data['move'] = data['Close'] - data['Close'].shift(1)
    data['up'] = np.where(data['move'] > 0, data['move'], 0)
    data['down'] = np.where(data['move'] < 0, data['move'], 0)
    data['average_gain'] = data['up'].rolling(period).mean()
    data['average_loss'] = data['down'].abs().rolling(period).mean()
    data['relative_strength'] = data['average_gain'] / data['average_loss']
    return 100.0 - (100.0 / (1.0 + data['relative_strength']))

'''
Function to get rsi and trend signals to later be used as the features 
Also dictates the direction of the stock which will be the target variable (prediction varible)
'''
def construct_signals(data, ma_period=60, rsi_period=14):
    data['SMA'] = data['Close'].rolling(window=ma_period).mean()
    # these are the 2 features
    data['trend'] = (data['Open'] - data['SMA']) * 100
    data['RSI'] = calculate_rsi(data, rsi_period) / 100
    # we need the target variables (labels)
    data['direction'] = np.where(data['Close'] - data['Open'] > 0, 1, -1)


if __name__ == '__main__':
    # Date range
    start_date = datetime.datetime(2018, 1, 1)
    end_date = datetime.datetime(2020, 1, 1)

    currency_data = fetch_data('TM', start_date, end_date)
    construct_signals(currency_data)

    currency_data = currency_data.dropna()

    X = currency_data[['trend', 'RSI']]
    y = currency_data['direction']

    # Create train test splits (20% test set size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and fit the model
    svm = SVC()
    svm.fit(X_train, y_train)

    # Perform grid search to find optimized parameters for model
    parameters = {'gamma': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    grid = list(ParameterGrid(parameters))

    best_accuracy = 0
    best_parameter = None

    for p in grid:
        svm = SVC(C=p['C'], gamma=p['gamma'])
        svm.fit(X_train, y_train)
        predictions = svm.predict(X_test)

        if accuracy_score(y_test, predictions) > best_accuracy:
            best_accuracy = accuracy_score(y_test, predictions)
            best_parameter = p

    print(best_parameter)

    # Reassign model with best parameters
    model = SVC(C=best_parameter['C'], gamma=best_parameter['gamma'])
    model.fit(X_train, y_train)
    # Make predictions on test set
    predictions = model.predict(X_test)

    print('Accuracy of the model: %.2f' % accuracy_score(y_test, predictions))
    print(confusion_matrix(predictions, y_test))
