import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import yfinance as yf
import plotly.graph_objects as go
import ta
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


def download_data(ticker, start, end):
    data = {}
    for tk in ticker:
        data[tk] = yf.download(tk, start, end)
    return data


def split_data(scaled_data):
    y = scaled_data.iloc[:, 3]
    X = scaled_data.drop(columns=[scaled_data.columns[3]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_ensemble_model(data_split):
    t1 = time.perf_counter()
    x_train, x_test, y_train, y_test = data_split
    cols = x_train.shape[1]
    rows = x_train.shape[0]
    rows_test = x_test.shape[0]
    cols_test = x_test.shape[1]
    input_shape = (cols, 1)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train_lstm = x_train.reshape((rows, cols, 1))
    x_test_lstm = x_test.reshape((rows_test, cols_test, 1))

    lstm_model = KerasRegressor(build_fn=create_lstm_model, input_shape=input_shape, epochs=100, verbose=0)
    lstm_model.fit(x_train_lstm, y_train)
    lstm_pred = lstm_model.predict(x_test_lstm)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)
    rf_pred = rf_model.predict(x_test)

    combined_pred = 0.5 * lstm_pred + 0.5 * rf_pred

    mse = mean_squared_error(y_test, combined_pred)
    rmse = np.sqrt(mse)
    mas = mean_absolute_error(y_test, combined_pred)
    t2 = time.perf_counter()

    print("Mean Squared Error : ", mse)
    print("Root Mean Squared Error : ", rmse)
    print("Mean Absolute Error : ", mas)
    print(f'Process time: {(t2 - t1) / 60:0.2f} minute(s)...')
    print("-----------------------------------------------")
    print("-----------------------------------------------")


def function_1(data):
    train_ensemble_model(data)


def main():
    tickers = ['JPM', 'BAC', 'ACN', 'NVDA']
    start = '2000-01-01'
    end = '2024-01-01'
    
    data = download_data(tickers, start, end)
    print('Data download Completed')
    
   
    for key in data:
        df = data[key]
        df['Date'] = df.index
        df = df.reset_index(drop=True)
        df['Daily Return'] = df['Close'].pct_change()
        df['Price Range'] = df['High'] - df['Low']
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df = df.drop(["Date", "Daily Return"], axis=1)
        df['RSI'] = df['RSI'].fillna(df['RSI'].mean())
        data[key] = df
    print('Creation of some features')
    
    scaled_data = {}
    for key in data:
        scaler = StandardScaler()
        scaled_data[key] = pd.DataFrame(scaler.fit_transform(data[key]), columns=data[key].columns)
    print('Scaling')
    
    data_split = []
    for key in scaled_data:
        data_split.append(split_data(scaled_data[key]))
    print('Data split')

    print('Entering Multiprocessing')
    try:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(function_1, data_split[i]) for i in range(len(data_split))]
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





## Better way to download ticker data

# import yfinance as yf
# import pandas_datareader.data as pdr
# yf.pdr_override()
# df = pdr.get_data_yahoo(name_stock, start="2023-01-01", end="2023-08-01")
# print(df.head())




























