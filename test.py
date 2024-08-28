import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import plotly.graph_objects as go
import ta
import time
from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import warnings
warnings.simplefilter('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import ast
with open('filter1_stocks.txt', 'r') as file:
    data = file.read()

tickers = data.splitlines()
print('all tickers = ', tickers)

actual_vector = []
req_ticker = []
pred_vector = []
# print("actual vector = ", actual_vector)

file_path = 'fiter2_stocks.txt'

for i in range(0, len(tickers)):
    actual_vector.append(yf.download(tickers[i], '2024-08-27', '2024-08-28')['Close'][0])

# print("actual price vector = ", actual_vector)
# print("actual price vector created")



# print("---------now defining the required functions------------")

def download_data(ticker, train_start, train_end, test_start, test_end):
    traindata = {}
    testdata = {}
    for tk in ticker:
        traindata[tk] = yf.download(tk, train_start, train_end) ## Correction
        testdata[tk] = yf.download(tk, test_start, test_end)
    return traindata, testdata

def split_data(scaled_traindata, scaled_testdata):
    scaler = StandardScaler()
    y_train = scaled_traindata.iloc[:, 3]
    y_test = scaled_testdata.iloc[:, 3]
    x_train = scaled_traindata.drop(columns = [scaled_traindata.columns[3]], axis = 1)
    x_test = scaled_testdata.drop(columns = [scaled_testdata.columns[3]], axis = 1)

    x_train = pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test),columns=x_test.columns)
    
    return x_train, x_test, y_train, y_test

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


# print("defining required ticker and predicted value vector")
def train_ensemble_model(data_split,name):
    # print('Entering train ensemble')
    t1 = time.perf_counter()
    x_train, x_test, y_train, y_test = data_split
    x_test.fillna(0,inplace=True) # Correction
    cols = x_train.shape[1]
    rows = x_train.shape[0]
    rows_test = x_test.shape[0]
    cols_test = x_test.shape[1]
    input_shape = (cols, 1)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train_lstm = x_train.reshape((rows, cols, 1))
    x_test_lstm = x_test.reshape((rows_test, cols_test, 1))

    # print('Lstm section')
    lstm_model = KerasRegressor(build_fn=create_lstm_model, input_shape=input_shape, epochs=10, verbose=0)
    # print('Entering lstm fiting')
    lstm_model.fit(x_train_lstm, y_train)
    # print('exit fit')
    # print(lstm_model.predict(x_train_lstm))
    lstm_pred = lstm_model.predict(x_test_lstm)
    # print(lstm_pred)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)
    rf_pred = rf_model.predict(x_test)

    combined_pred = 0.5 * lstm_pred + 0.5 * rf_pred
    print('combined predicted value : ',combined_pred.mean())
    pred_vector.append(combined_pred)
    # print(pred_vector)
    t2 = time.perf_counter()

    print(tickers[name])
    print('predicted value = ', combined_pred.mean() , ' actual value = ', actual_vector[name])

    if (combined_pred.mean() - actual_vector[name])/actual_vector[name] > 0.025:
        print(f'This is the one {tickers[name]}')
        with open(file_path, 'a') as file:
            file.write(tickers[name])
            file.write('\n')

    print(f'Process time: {(t2 - t1) / 60:0.2f} minute(s)...')
    print("-----------------------------------------------")
    print("-----------------------------------------------")

def function_1(data,name):
    train_ensemble_model(data,name)


def main():
    trainstart = '2000-01-01'
    trainend = '2024-08-26'

    teststart = '2024-08-26'
    testend = '2024-08-28'

    traindata, testdata = download_data(tickers, trainstart, trainend, teststart, testend)
    # print('Data download Completed')


    # print("creating features")

    for trainkey in traindata:
        df = traindata[trainkey]
        df['Date'] = df.index
        df = df.reset_index(drop=True)
        df['Price Range'] = df['High'] - df['Low']
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df = df.drop(["Date"], axis=1)
        df['RSI'] = df['RSI'].fillna(df['RSI'].mean())
        traindata[trainkey] = df

    for testkey in testdata:
        df = testdata[testkey]
        df['Date'] = df.index
        df = df.reset_index(drop=True)
        df['Price Range'] = df['High'] - df['Low']
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df = df.drop(["Date"], axis=1)
        df['RSI'] = df['RSI'].fillna(df['RSI'].mean())
        testdata[testkey] = df

    # print('Creation of some features')
    scaled_traindata = {}
    scaled_testdata = {}

    for trainkey in traindata:                  
        scaled_traindata[trainkey] = pd.DataFrame((traindata[trainkey]),
                                                  columns=traindata[trainkey].columns)
        scaled_testdata[trainkey] = pd.DataFrame((testdata[trainkey]),
                                                 columns=testdata[trainkey].columns)

    # print('Scaling')

    data_split = []

    for trainkey, testkey in zip(scaled_traindata, scaled_testdata):
        data_split.append(split_data(scaled_traindata[trainkey], scaled_testdata[testkey]))
    # print('Data split done')

    print('Entering Multiprocessing')
    try:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(function_1, data_split[i],i) for i in range(len(data_split))]
            results = [future.result() for future in futures]
    except concurrent.futures.process.BrokenProcessPool as e:
        print(f"BrokenProcessPool error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print('Exiting Multiprocessing')


if __name__ == "__main__":
    t1 = time.perf_counter()
    main()
    t2 = time.perf_counter()

    print(f'Total time: {(t2 - t1) / 60:0.2f} minute(s)...')


