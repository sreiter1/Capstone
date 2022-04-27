# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 08:14:11 2022

@author: sreit
"""


import datetime



class conversionTables:
    tickerConversionTable =  {"time"     : "daily_adjusted",
                              "balance"  : "balance_sheet",
                              "cash"     : "cash_flow",
                              "earnings" : "earnings",
                              "overview" : "fundamental_overview",
                              "income"   : "income_statement"}
        
    # Converts user unput to the columns in the table.  Provides a filter to 
    # prevent database corruption.
    dailyConversionTable =  {"OPEN"         : "open",
                             "CLOSE"        : "close",
                             "HIGH"         : "high",
                             "LOW"          : "low",
                             "ADJCLOSE"     : "adj_close",
                             "VOLUME"       : "volume",
                             "SPLIT"        : "split",
                             "ADJRATIO"     : "adjustment_ratio",
                             "MA20"         : "mvng_avg_20",
                             "MA50"         : "mvng_avg_50",
                             "MACD12"       : "macd_12_26",
                             "MACD19"       : "macd_19_39",
                             "VOL20"        : "vol_avg_20",
                             "VOL50"        : "vol_avg_50",
                             "OBV"          : "on_bal_vol",
                             "DAYCHANGE"    : "percent_cng_day",
                             "TOTALCHANGE"  : "percent_cng_tot",
                             "RSI"          : "rsi"}




class validationFunctions:
    
    # confirms that the value is a boolean
    def validateBool(self, value):
        if not (isinstance(value, bool)):
            raise ValueError("Input must be of type 'bool'.")
        else:
            return 1
    
    # confirms that the ticker list is a python list of text strings
    def validateListString(self, inputList = []):
        if not isinstance(inputList, list):
            raise TypeError("Input not a list")
        elif inputList == []:
            raise ValueError("Input list is empty")
        if not all(isinstance(item, str) for item in inputList):
            raise TypeError("Input not a list of strings.\n")
        
        return 1
    
    
    def validateListInt(self, inputList = []):
        if not isinstance(inputList, list):
            raise TypeError("Input not a list")
        elif inputList == []:
            raise ValueError("Input list is empty")
        if not all(isinstance(item, int) for item in inputList):
            raise TypeError("Input not a list of integers.\n")
        
        return 1
    
    
    def validateListFloat(self, inputList = []):
        if not isinstance(inputList, list):
            raise TypeError("Input not a list")
        elif inputList == []:
            raise ValueError("Input list is empty")
        if not all((isinstance(item, float) or isinstance(item, int)) for item in inputList):
            raise TypeError("Input not a list of numeric.\n")
        
        return 1
    
    
    def validateDateString(self, date_text):
        # Checks the string entered to see if it can be parsed into a date.
        try:
            datetime.datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYY-MM-DD")
            
        return 1
    
    
    def validateString(self, string):
        if not isinstance(string, str):
            raise TypeError("Input not a string.\n")
        else:
            return 1
        
        
    def validateInteger(self, integer):
        if not isinstance(integer, int):
            raise TypeError("Input is not an integer.\n")
        else:
            return 1
        
        
    def validateNum(self, number):
        if not isinstance(number, float):
            raise TypeError("Input is not an integer.\n")
        else:
            return 1
            
            
            