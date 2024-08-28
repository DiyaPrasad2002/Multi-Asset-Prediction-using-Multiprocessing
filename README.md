# Multi Asset Prediction Using Multiprocessing

## Aim of the project : 
The aim of the project was to study multiple methods of analysing and evaluating the stock price movements, alongside understanding the concept of multi-processing. Multi-processing enables parallel execution of tasks by using multiple CPU cores to get the tasks done in parallel. Each task runs independently on a different CPU core. While, LSTMs and Random Forest algorithms are computationally intensive and sequentially applying them to all the stocks might not be very computationally feasible and time efficient, multi-processing makes it more efficient by training multiple models simultaneously and using multiple CPU cores.

## Methods to Analyse the Stock Prices : 
1. Analysing Open and Close Price movements over a period of time
2. Analysing RSI - Relative Strength Index
3. Analysing EMA - Exponential Moving Average
4. Studying the daily returns of the stock

## Dataset Used to Make Predictions : 
In order to study the stock price movements, we make use of the yfinance data via python. In the notebook, the training of the model occurs sequentially and the time taken is against the time that would be used via multiprocessing. While using 4 stocks prices over a span of 24 years, sequential training of the model takes more than 10 minutes to train and make predictions. However, training multiple models simultaneously takes more than 10 minutes, however, multi-processing takes close to 4 minutes only.

![4stocks](https://github.com/user-attachments/assets/60ab44e9-fada-4efb-8d8f-9eed0a61952a)

## Filtering the Stocks to Make Predictions On : 
In order to filter out which stocks to invest on, we take the estimate percentage increase of the stock price in the last 5 minutes of closing of the market. We set the threshold of the increase in the value to 0.25%. This is performed in order to look at the volatility of the stock and its movement, which is expected to be followed. After the first list of stocks is filtered out, the models are fit on the filtered stocks, which are expected to give a good return. 

Out of the stocks on which the models are trained and price are predicted, if the stocks give a return over a pre-set threshold, the stock is then chosen to be invested in. 

## Forecasting the Stock Prices of the Stocks Listed on NSE
In order to get an overview of the leverage of multi-processing, the same model training was employed on all the filtered stocks, after the first filtering stage, that are listed on the NSE. A threshold value for the percentage increase in the stock price was established in order to filter out those stocks which were predicted to show the maximum rise in the prices for the next day or for the a particular window period. 

It was found that there were 259 stocks in total, out of which 36 were filtered out, which consumed around 0.65 minutes/stock for training and price prediction. 
Average time for sequential price prediction of all stocks and filtering = 0.65*36 ~ 40.95 minutes
Time taken through multi-processing = 3.72 minutes

![Screenshot 2024-08-28 153852](https://github.com/user-attachments/assets/d5b8d148-7311-4e15-94e0-f0f83325933c)


![Screenshot 2024-08-28 153837](https://github.com/user-attachments/assets/647b48a8-1c4b-4f34-a079-9a41cd8b6145)








