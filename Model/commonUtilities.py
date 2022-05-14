# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 08:14:11 2022

@author: sreit
"""


import datetime
import psutil
import os
import pandas as pd
import time



class conversionTables:
    tickerConversionTable =  {"alphatime"   : "alpha_daily",
                              "balance"      : "balance_sheet",
                              "cash"         : "cash_flow",
                              "earnings"     : "earnings",
                              "overview"     : "fundamental_overview",
                              "income"       : "income_statement"}
        
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
                             "BOLLINGER20"  : "bollinger_20",
                             "TP20"         : "tp20",
                             "BOLLINGER50"  : "bollinger_50",
                             "TP50"         : "tp50",
                             "MACD12"       : "macd_12_26",
                             "MACD19"       : "macd_19_39",
                             "VOL20"        : "vol_avg_20",
                             "VOL50"        : "vol_avg_50",
                             "OBV"          : "on_bal_vol",
                             "DAYCHANGE"    : "percent_cng_day",
                             "TOTALCHANGE"  : "percent_cng_tot",
                             "RSI"          : "rsi"}
    
    def loadStockListCSV(self, stockListFileName, saveToDB = True):
        # reads a csv file of stock tickers and optionally saves them to 
        # 'ticker_symbol_list' table in the database.
        try:
            # Open the csv-based list of tickers/companies
            stockFile = open(stockListFileName, "r") 
            
        except:
            print("Bad stock list file.  Unable to open.")
            return 1
        
        # read each line and create a list for the outputs
        Lines = stockFile.readlines()
        
        DF_ticker = [] # ticker symbol list
        DF_name = [] # name of the company
        DF_exchange = [] # exchange that the stock is traded on
        DF_recordDate = [] # date that the ticker was added to the database
        
        # open each line, split on the comma to get each value, append the 
        # ticker, name, and exchange from the CSV to the lists, and add today's
        # date to the record date.  
        for line in Lines:
            stock = line.split(",")
            DF_ticker.append(stock[0])
            DF_name.append(stock[1])
            DF_exchange.append(stock[2].strip('\n'))
            DF_recordDate.append(datetime.date.today())
            
            # execute a save to the 'ticker_symbol_list' table.
            if saveToDB:
                self._updateTickerList(ticker_symbol = stock[0],
                                       name = stock[1],
                                       exchange = stock[2].strip("\n"),
                                       recordDate = str(datetime.date.today()))
            
        # create the dataframe with all the recorded data
        df = pd.DataFrame([DF_ticker,
                           DF_recordDate,
                           DF_name,
                           DF_exchange])
        
        # label the data
        df.index = ["ticker_symbol", "recordDate", "name", "exchange"]
        df = df.transpose()
        
        # return the data
        return df
    



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
        if not (isinstance(number, float) or isinstance(number, int)):
            raise TypeError("Input is not an integer.\n")
        else:
            return 1
            
            



class callLimitExceeded(Exception):
    # raised when the program detects that the call limit was exceeded, 
    # either because the limit set in 'getData' is exceeded or if the API 
    # returns a specific error.
    pass

class vpnResetFailed(Exception):
    # Occurs if the VPN reset function fails.  This is likely to be triggered
    # after the API call limit is exceeded if the VPN cannot be reset.
    pass



class VPNProcessTools:
    
    def _checkIfProcessRunning(self, processName):
        # Check if there is any running process that contains the given name 
        # processName.  
        
        #Iterate over the all the running process
        for proc in psutil.process_iter():
            try:
                # Check if process name contains the given name string.
                if processName.lower() in proc.name().lower():
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False;
    
    
    
    def resetVPN(self):
        print("\n\n   Stopping VPN...")
        process = self._checkIfProcessRunning("nsv")  # look for the Norton VPN service
        while process != False: # keep looking until the service is found
            vpnKillStatus = os.system("schtasks /run /tn killVPN")
            # stops the norton vpn I am using. The command line needs
            # run "taskkill /f /im NSV.exe" as an administrator.  This 
            # command is contained in a batch file that is then added to
            # to the windows scheduler as an "on demand" task.  This task
            # is then executed via the line above to avoid the need for
            # the user (i.e. me) to provide permission for the batch file
            # to run.  The output is assigned to 'vpnKillStatus', which 
            # is 0 if completed successfully and 1 otherwise.
            
            if vpnKillStatus == 1:
                raise vpnResetFailed("\nScheduled Task failed to complete successfully.\n")
            
            # This opens a security vulnerability if the batch file is altered.
            time.sleep(3) # helps to keep from confusing the VPN.  3 was pulled from thin air.
            process = self.tools._checkIfProcessRunning("nsv") # look for the Norton VPN service
        
        
        print("   Restarting VPN...")
        while process == False:  # if the Norton VPN process doeesn't exist (not runninng), 'process' will be false
            # Call the OS to start Norton VPN.  The VPN is setup to automatically connect to the servers.
            os.system("C:\\Program Files\\NortonSecureVPN\\Engine\\5.1.1.5\\nsvUIStub.exe")
            time.sleep(5)
            process = self._checkIfProcessRunning("nsv")
        
        print("   Resetting API call counter...")
        # reset the API calls and call times.
        self._resetTotalApiCalls()
        apiCallTime = time.time() - 60
        self._apiRecentCallTimes = [apiCallTime for i in range(self._rate)]
        print("   API call counter reset.")
        print("   VPN restart complete.\n\n")
        
        return 0    
    
    
    

    