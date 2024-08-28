import ast
import pandas as pd
import numpy as np
import yfinance as yf


def value_change(df):
    present = 1
    past = 5
    val_present = df.iloc[-present]
    val_past = df.iloc[-past]
    print(val_present, val_past)
    val = (val_present - val_past)/val_past
    return val


def download_data(name, start_date, end_date, interval):
    data = yf.download(name, start = start_date, end = end_date, interval = interval)
    data['DateTime'] = data.index
    data = data.reset_index(drop = True)
    return data


def stocks_to_watch(name, pct):
    #find the data
    data = download_data(name, start_date, end_date, interval)
    df = data['Close']
    #find percentage change
    val = value_change(df)

    if val>pct:
        return True
    else:
        return False
    

interval = '1m'
start_date = '2024-08-22'
end_date = '2024-08-27'

with open('Stocks.txt', 'r') as file:
    data = file.read()
    
tickers = ast.literal_eval(data)
#print(tickers)


filtered_tickers = []
pct = 0.0025

file_path = 'filter1_stocks.txt'

for tk in tickers:
    if(stocks_to_watch(tk, pct) == True):
        with open(file_path, 'a') as file:
            file.write(tk)
            file.write('\n')
