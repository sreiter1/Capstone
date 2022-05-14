# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:21:21 2022

@author: sreit
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import statsmodels.api as sm
import warnings
import sqlite3
import commonUtilities
import analysis2

import datetime as dt
import os
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler



warnings.filterwarnings('ignore')




class models:
    def __init__(self, dataBaseSaveFile = "./stockData.db", splitDate = "2020-01-01"):
        self.DB = sqlite3.connect(dataBaseSaveFile)
        self._cur = self.DB.cursor()
        self._tickerList = []   # Empty list that gets filled with a list of tickers to be considered
        self._data = pd.DataFrame()
        self.tradingDateSet = []  # List of dates in YYYY-MM-DD format that are trading dates in the database
        self.dailyTableNames = ["alpha", "yahoo"]
        self.splitDate = pd.to_datetime(splitDate)
        
        self.validate = commonUtilities.validationFunctions()
        
        # Converts a user string to the names of tables, and for 'ticker_symbol_list'
        self._tickerConversionTable = commonUtilities.conversionTables.tickerConversionTable
        
        # Converts user unput to the columns in the table.  Provides a filter to 
        # prevent database corruption.
        self._dailyConversionTable = commonUtilities.conversionTables.dailyConversionTable
        
        self.indicatorList = {"MA20":        "mvng_avg_20", 
                              "MA50":        "mvng_avg_50", 
                              "MACD12":      "macd_12_26", 
                              "MACD19":      "macd_19_39",
                              "OBV":         "on_bal_vol", 
                              "RSI":         "rsi",
                              "BOLLINGER20": "bollinger_20",
                              "BOLLINGER50": "bollinger_50"}
        self.analysis = analysis2.analysis()
        
        
        
        def getTickers(self):
            data = self.analysis.filterStocksFromDataBase(dailyLength = 1250, 
                                                          maxDailyChange = 50, 
                                                          minDailyChange = -50, 
                                                          minDailyVolume = 50000)
            
            self._tickerList = list(data["ticker_symbol"])
            data['recordDate'] = pd.to_datetime(data['recordDate'])
            self._data = data
        
        
        
        def ARIMA(self):
            pass
        
        
        
        
        def LSTM(self):
            
            for ticker in self._tickerList:
                
                df = self._data.loc[self._data["ticker_symbol"] == ticker]
                
                high_prices = df['high']
                low_prices  = df['low']
                df['mid']  = [h+l/2.0 for h,l in zip(high_prices, low_prices)]
                
                
                train_data = df['mid'].loc[df["recordDate"] < self.splitDate].as_matrix()
                test_data  = df['mid'].loc[df["recordDate"] > self.splitDate].as_matrix()
                
                scaler = MinMaxScaler()
                train_data = train_data.reshape(-1,1)
                test_data  = test_data.reshape(-1,1)
                
                
                smoothing_length = 90
                slide_length = 30
                for window in range(0, len(train_data)-1, slide_length):
                    scaler.fit(train_data[window : window + smoothing_length  ,:])
                    fitted_data = scaler.transform(train_data[window : window + smoothing_length  ,:])
                    
                    
                    
                
                # You normalize the last bit of remaining data
                scaler.fit(train_data[di+smoothing_window_size:,:])
                train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])
                
                
            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        