# Multi Asset Prediction Using Multiprocessing

## Aim of the project : 
The aim of the project was to study multiple methods of analysing and evaluating the stock price movements, alongside understanding the concept of multi-processing. Multi-processing enables parallel execution of tasks by using multiple CPU cores to get the tasks done in parallel. Each task runs independently on a different CPU core. While, LSTMs and Random Forest algorithms are computationally intensive and sequentially applying them to all the stocks might not be very computationally feasible and time efficient, multi-processing makes it more efficient by training multiple models simultaneously and using multiple CPU cores.

# Methods to Analyse the Stock Prices : 
1. Analysing Open and Close Price movements over a period of time
2. Analysing RSI - Relative Strength Index
3. Analysing EMA - Exponential Moving Average
4. Studying the daily returns of the stock

# Dataset Used to Make Predictions : 
In order to study the stock price movements, we make use of the yfinance data via python. In the notebook, the training of the model occurs sequentially and the time taken is against the time that would be used via multiprocessing. While using 4 stocks prices over a span of 24 years, sequential training of the model takes more than 10 minutes to train and make predictions. However, training multiple models simultaneously takes more than 10 minutes, however, multi-processing takes close to 4 minutes only.

![4stocks](https://github.com/user-attachments/assets/60ab44e9-fada-4efb-8d8f-9eed0a61952a)



