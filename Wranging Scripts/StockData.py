# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:54:23 2021

getData.checkForErrors has a bad SQL fuction with the use of 'tickerErrAttr'
and getData.saveToSQL in the cmd_string with 'table_name' as a function argument
as the column value to call in the SQL request.  Need better option...

@author: sreit
"""

import pandas as pd
import datetime
import time
import requests
import sqlite3
import os
import sys



class getData:
    def __init__(self, API_KEY = "NYLUQRD89OVSIL3I", rate = 5, limit = 500, dataBaseSaveFile = "SQLiteDB/stockData.db"):
        self.DB = database(dataBaseSaveFile)
        self.rate = rate   # calls per minute
        self.limit = limit  # calls per day
        self.apiTotalCalls = 0  # number of calls already made
        self.API_KEY = API_KEY  # Key for use with alphaVantage API
        self.alphaVantageBaseURL = "https://www.alphavantage.co/query?"
        self.tickerList = []    # Empty list of stock tickers
        self.apiCallTime = time.time() - 60/self.rate
        self.workingFileText = ""
        


    def loadStockListCSV(self, stockListFileName, saveToSQL = True):
        try:
            # Open the csv-based list of tickers/companies
            stockFile = open(stockListFileName, "r") 
            
        except:
            print("Bad stock list file.  Unable to open.")
            return 1
        
        Lines = stockFile.readlines()
        
        DF_ticker = []
        DF_name = []
        DF_exchange = []
        DF_recordDate = []
        
        for line in Lines:
            stock = line.split(",")
            DF_ticker.append(stock[0])
            DF_name.append(stock[1])
            DF_exchange.append(stock[2].strip('\n'))
            DF_recordDate.append(datetime.date.today())
            
            
        df = pd.DataFrame([DF_ticker,
                           DF_name,
                           DF_exchange,
                           DF_recordDate])
        
        df.index = ["ticker_symbol", "name", "exchange", "recordDate"]
        df = df.transpose()
        
        if saveToSQL:
            self.saveToSQL(df, "ticker_symbol_list")
        
        return df
    
    
    
    def checkApiCalls(self):
        self.apiTotalCalls += 1
        
        if self.apiTotalCalls > self.limit:
            raise SystemExit("Daily API call limit met.  Please wait or increase limit.")
        
        
        while time.time() < self.apiCallTime + 60/self.rate:
            timeRemaining = int(self.apiCallTime + 60/self.rate - time.time())
            if(60/self.rate > 10):
                print("\rDelay for API call rate.  Time remaining = " + str(timeRemaining) + "            ", end = "\r")
            time.sleep(1)
        
        print("\rNumber of API calls today = " + str(self.apiTotalCalls) + "               ")
        self.apiCallTime = time.time()
        
    
    
    def checkForErrors(self, ticker_symbol, jsonResponse, funcName = "", tickerErrAttr = ""):
        jsonKeys = list(jsonResponse.keys())
        
        if jsonKeys == []:
            msgType = "Empty"
            message = "API returned no information."
        else:
            msgType = jsonKeys[0]
            message = jsonResponse[msgType]
            
        if "standard API call frequency is 5 calls per minute and 500 calls per day" in message:
            print("API calls exhausted or too frequent.")
            raise ValueError("API calls exhausted for today, or calls are too frequent.")
        
        if msgType in ["Error Message", "Note", "Information", "Empty"]:
            sqlString  = "INSERT INTO errors_tracker \n"
            sqlString += "('ticker_symbol', 'recordTime', 'recordDate', " + \
                         "'errorType', 'errorSource', 'errorMessage')\n"
            sqlString += "VALUES(?, ?, ?, ?, ?, ?);\n"
            
            argList = (ticker_symbol, str(time.time()), str(datetime.date.today()), msgType, funcName, message)
                       
            cur = self.DB.stockDB.cursor()
            cur.execute(sqlString, argList)
            
            if tickerErrAttr != "":
                sqlString  = "UPDATE ticker_symbol_list \n"
                sqlString += "SET '" + tickerErrAttr + "' = 1, \n"
                sqlString += "    'data_" + tickerErrAttr[6:] + "' = 'Err' \n"
                sqlString += "WHERE ticker_symbol = ? ; \n"
                
                argList = (ticker_symbol,)
                cur.execute(sqlString, argList)
            
            self.DB.stockDB.commit()
            
            print("Error on ticker '" + ticker_symbol + "' and function '" + funcName + "':  " + message)
            
            return 1
        else:
            if tickerErrAttr != "":
                cur = self.DB.stockDB.cursor()
                cmd_string = "UPDATE ticker_symbol_list \n" + \
                             "SET '" + tickerErrAttr + "' = 0, \n" + \
                             "    'data_" + tickerErrAttr[6:] + "' = ? \n" + \
                             "WHERE ticker_symbol = ? ;\n"
                
                argList = (str(datetime.date.today()), ticker_symbol)
                
                cur.execute(cmd_string, argList)
                self.DB.stockDB.commit()
        
        return 0
    
    
    
    def getTimeSeriesDaily(self, ticker_symbol, save = True):
        # Collects the time Series data for a selected ticker
                    
        # Build the request URL for alphaVantage API
        requestURL = self.alphaVantageBaseURL + "function=TIME_SERIES_DAILY_ADJUSTED&" + \
                     "outputsize=full&symbol=" + ticker_symbol + "&apikey=" + self.API_KEY
        
        # Send API request
        self.checkApiCalls()
        
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self.checkForErrors(ticker_symbol, data, "Time Series Daily", "error_daily_adjusted"):
            return None 
        
        # Extract the data from the response and convert it to a pandas dataframe
        stockDF = pd.DataFrame(data["Time Series (Daily)"])
        stockDF = stockDF.transpose()
        
        stockDF.rename({"1. open": "open",
                        "2. high": "high",
                        "3. low": "low",
                        "4. close": "close",
                        "5. adjusted close": "adj_close",
                        "6. volume": "volume",
                        "7. dividend amount": "dividend",
                        "8. split coefficient": "split"}, axis=1, inplace=True)
        
        stockDF['ticker_symbol'] = ticker_symbol
        stockDF['recordDate'] = stockDF.index
        
        if save:
            self.saveToSQL(stockDF, "daily_adjusted", ticker_symbol)
            
            cmd_string = "UPDATE ticker_symbol_list \n" + \
                         "SET 'hist_length' = ? \n" + \
                         "WHERE ticker_symbol = ?;\n"
            
            argList = (len(stockDF.index), ticker_symbol)
            
            cur.execute(cmd_string, argList)
            self.DB.stockDB.commit()
                
        return stockDF
    
    

    def getFundamentalOverview(self, ticker_symbol, save = True):
        # Collects the time Series data for a selected ticker
                    
        # Build the request URL for alphaVantage API
        requestURL = self.alphaVantageBaseURL + "function=OVERVIEW&" + \
                     "symbol=" + ticker_symbol + "&apikey=" + self.API_KEY
        
        # Send API request
        self.checkApiCalls()
        
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self.checkForErrors(ticker_symbol, data, "Fundamental Overview", "error_fund_overview"):
            return None 
        
        # Extract the data from the response and convert it to a 
        stockDF = pd.DataFrame(list(data.items()), index = data.keys(), columns = ["Label","Data"])
        
        stockDF.rename(index={"Symbol":"ticker_symbol", \
                              "52WeekHigh":"high_52week", \
                              "52WeekLow":"low_52week", \
                              "50DayMovingAverage":"moving_average_50d", \
                              "200DayMovingAverage":"moving_average_200d"}, inplace=True)
        
        stockDF = stockDF.transpose()
        stockDF.drop('Label', inplace = True)
        
        stockDF['recordDate'] = str(datetime.date.today())
        
        if save:
            self.saveToSQL(stockDF, "fund_overview", ticker_symbol)
            
        return stockDF
    


    def getCashFlow(self, ticker_symbol, save = True):
        # Collects the time Series data for a selected ticker
                    
        # Build the request URL for alphaVantage API
        requestURL = self.alphaVantageBaseURL + "function=CASH_FLOW&" + \
                     "symbol=" + ticker_symbol + "&apikey=" + self.API_KEY
        
        # Send API request
        self.checkApiCalls()
        
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self.checkForErrors(ticker_symbol, data, "Cash Flow", "error_cash_flow"):
            return None 
        
        # Extract the data from the response and convert it to a dataframe
        annual = pd.DataFrame(data['annualReports'])
        quarterly = pd.DataFrame(data['quarterlyReports'])
        
        # Add boolean column for annual vs quarterly and for the ticker symbol
        annual['annualReport'] = 1
        quarterly['annualReport'] = 0
        annual['ticker_symbol'] = ticker_symbol
        quarterly['ticker_symbol'] = ticker_symbol
        
        # concatenate the quarterly and annual dataframes
        frames = [annual, quarterly]
        stockDF = pd.concat(frames)
        
        stockDF.rename({"proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet":
                        "proceedsFromIssuanceOfLongTermDebt", \
                        "fiscalDateEnding":"recordDate"}, axis=1, inplace=True)
        
        if save:
            self.saveToSQL(stockDF, "cash_flow", ticker_symbol)
            
        return stockDF
    
    
    
    def getIncomeStatement(self, ticker_symbol, save = True):
        # Collects the time Series data for a selected ticker
                    
        # Build the request URL for alphaVantage API
        requestURL = self.alphaVantageBaseURL + "function=INCOME_STATEMENT&" + \
                     "symbol=" + ticker_symbol + "&apikey=" + self.API_KEY
        
        # Send API request
        self.checkApiCalls()
        
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self.checkForErrors(ticker_symbol, data, "Income Statement", "error_income_statement"):
            return None 
        
        # Extract the data from the response and convert it to a dataframe
        annual = pd.DataFrame(data['annualReports'])
        quarterly = pd.DataFrame(data['quarterlyReports'])
        
        # Add boolean column for annual vs quarterly and for the ticker symbol
        annual['annualReport'] = 1
        quarterly['annualReport'] = 0
        annual['ticker_symbol'] = ticker_symbol
        quarterly['ticker_symbol'] = ticker_symbol
        
        # concatenate the quarterly and annual dataframes
        frames = [annual, quarterly]
        stockDF = pd.concat(frames)
        
        stockDF.rename({"fiscalDateEnding":"recordDate"}, axis=1, inplace=True)
        
        if save:
            self.saveToSQL(stockDF, "income_statement", ticker_symbol)
            
        return stockDF



    def getEarnings(self, ticker_symbol, save = True):
        # Collects the time Series data for a selected ticker
                    
        # Build the request URL for alphaVantage API
        requestURL = self.alphaVantageBaseURL + "function=EARNINGS&" + \
                     "symbol=" + ticker_symbol + "&apikey=" + self.API_KEY
        
        # Send API request
        self.checkApiCalls()
        
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self.checkForErrors(ticker_symbol, data, "Earnings", "error_earnings"):
            return None 
        
        # Extract the data from the response and convert it to a dataframe
        stockDF = pd.DataFrame(data['quarterlyEarnings'])
        
        # Add boolean column for annual vs quarterly and for the ticker symbol
        stockDF['ticker_symbol'] = ticker_symbol
        
        stockDF.rename({"fiscalDateEnding":"recordDate"}, axis=1, inplace=True)
        
        if save:
            self.saveToSQL(stockDF, "earnings", ticker_symbol)
            
        return stockDF



    def getBalanceSheet(self, ticker_symbol, save = True):
        # Collects the time Series data for a selected ticker
                    
        # Build the request URL for alphaVantage API
        requestURL = self.alphaVantageBaseURL + "function=BALANCE_SHEET&" + \
                     "symbol=" + ticker_symbol + "&apikey=" + self.API_KEY
        
        # Send API request
        self.checkApiCalls()
        
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self.checkForErrors(ticker_symbol, data, "Balance Sheet", "error_balance_sheet"):
            return None 
        
        # Extract the data from the response and convert it to a dataframe
        annual = pd.DataFrame(data['annualReports'])
        quarterly = pd.DataFrame(data['quarterlyReports'])
        
        # Add boolean column for annual vs quarterly and for the ticker symbol
        annual['annualReport'] = 1
        quarterly['annualReport'] = 0
        annual['ticker_symbol'] = ticker_symbol
        quarterly['ticker_symbol'] = ticker_symbol
        
        # concatenate the quarterly and annual dataframes
        frames = [annual, quarterly]
        stockDF = pd.concat(frames)
        
        stockDF.rename({"fiscalDateEnding":"recordDate"}, axis=1, inplace=True)
        
        if save:
            self.saveToSQL(stockDF, "balance_sheet", ticker_symbol)
            
        return stockDF
    


    def saveToSQL(self, stockDF, table_name, ticker_symbol = ""):
        # Saves a dataframe to the SQLite database.  Functions as a wrapper for
        # this class and its dataframes.  The dataframes and SQLite schema were
        # setup to match and work well together.
        
        # cursor for the database
        cur = self.DB.stockDB.cursor()
        
        print(ticker_symbol + " ", end = "")
        
        # Command string that identifies which table to insert data into.
        # "table_name" is passed into this function, which makes this function
        # less than ideally secure.  Column_string accumulates the columns from
        # the dataframe in a way that the SQLite database can identify.  
        # value_string accumulates '?' to include the values that each column 
        # contains.
        
        cmd_string = "INSERT INTO " + table_name + " \n"
        column_string = "('"
        value_string = "VALUES("
        
        for key in stockDF.keys():
            column_string += key + "', '"
            value_string  += "?, "
        
        # Close out the strings for the SQL transaction
        column_string = column_string[:-3] + ")\n"
        value_string  = value_string[:-2]  + ");\n"
        cmd_string += column_string + value_string
        
        
        # Convert the pandas dataframe into a list of tuples for addition to
        # the SQLite database, and commit the changes.
        data_list = list(stockDF.to_records(index=False))
        print("\r" + self.workingFileText + ticker_symbol.rjust(6) + " records written: " + str(len(data_list)).rjust(7), end = "")
        cur.executemany(cmd_string, data_list)
        self.DB.stockDB.commit()
        
        print("\nDone", end = "")
    
    
    
    def addPickleToTickerList(self, directory = "./"):
        print("Adding entries for stocks in pickle files...")
        
        # Read the database and see what trackers are already in the database.
        # allows for the insertion of new records if the record doesn't exist
        cur = self.DB.stockDB.cursor()
        query = cur.execute("SELECT * FROM ticker_symbol_list")
        cols = [column[0] for column in query.description]
        DF_tickerList = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
        DF_tickerList = list(DF_tickerList["ticker_symbol"])
        
        
        # Collect a list of file names in the directory holding the pickle files
        dirList = os.listdir(directory)
        
        DF_dirList = []
        
        for file in dirList:
            # Parse file names; files are named like this:
            #
            # 2_AAAA_Type_detailedType.pickle
            
            fileName = file.split(".")[0].split("_")
            DF_dirList.append(fileName[1])
        
        
        DF_dirList = list(dict.fromkeys(DF_dirList))
        DF_dirList = list(set(DF_dirList) - set(DF_tickerList))
        DF_dirList.sort()
        
        DF_tickers = pd.DataFrame([DF_dirList])
        DF_tickers = DF_tickers.transpose()
        DF_tickers["recordDate"] = str(datetime.date.today())
        
        
        DF_tickers.columns = ["ticker_symbol", "recordDate"]
        self.saveToSQL(DF_tickers, "ticker_symbol_list")
         
        print("Done.")
        
        return DF_tickers
    
    
    
    def copyPickleToSQL(self, stocksDirectory = "./"):
        # Process the pickle files generated from a previous script for 
        # inclusion into the SQLite database.  "Conversion" is a dictionary
        # translates the last part of the filename to the SQLite table
        # that the data needs to be entered into.
        
        conversion = {"TimeData":"daily_adjusted",
                      "annualBalance":"balance_sheet",
                      "quarterlyBalance":"balance_sheet",
                      "annualCash":"cash_flow",
                      "quarterlyCash":"cash_flow",
                      "quarterlyEarnings":"earnings",
                      "Overview":"fund_overview",
                      "annualIncome":"income_statement",
                      "quarterlyIncome":"income_statement"}
        
        self.addPickleToTickerList(stocksDirectory)
        
        # Collect a list of file names in the directory holding the pickle files
        dirList = os.listdir(stocksDirectory)
        
        # Total and count for progress monitor
        numberOfFiles = str(len(dirList)).rjust(6)
        count = 0
        
        # Itterate through all the .pickle files to sort the data into the SQLite
        # tables and entries.  The Schema included in the Database class matches
        # the values in the .pickle files, and by extension the data downloaded
        # from the AlphaVantage API.
        for file in dirList:
            # Keep track of overall progress
            count += 1
            self.workingFileText = "Working file " + str(count).rjust(6) + " of " + numberOfFiles + "      "
            
            
            # Parse file names; files are named like this:
            #
            # 2_AAAA_Type_detailedType.pickle
            #
            # The name can be split along the underscores to give:
            # 2: ordinal number of the ticker; basically an inventory number
            # AAAA: Stock ticker as listed on its exchange
            # Type: which API call generated the data
            # detailedType: API call, plus frequency
            
            typeFile = file.split(".")[0].split("_")[-1]
            ticker_symbol = file.split(".")[0].split("_")[1]
            
            
            # Read the pickle file into a pandas dataframe.
            df = pd.read_pickle(stocksDirectory + file)
            
            # String used later in the SQLite call; see details near the end
            # of this function.
            histLengthString = "\n"
            
            # Series of if statements that determine the which type of data is 
            # in the dataframe, followed by a series of adustments that are used
            # to prepare the dataframe for entry into the SQLite database.
            # 
            # Most of this section renames pandas series and includes basic data
            # for each record (ticker symbol, date of the information, latest 
            # date that the information was pulled for the ticker symbol, etc.)
            if typeFile == "TimeData":
                df.rename({"adjusted close": "adj_close",
                           "dividend amount": "dividend",
                           "split coefficient": "split"}, axis=1, inplace=True)
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                tickerTrackerEntry = "daily_adjusted"
                histLengthString = ", \n    'hist_length' = " + str(len(df.index)) + "\n"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "annualBalance":
                df['annualReport'] = 1
                df['ticker_symbol'] = ticker_symbol
                df["recordDate"] = df.index.date
                tickerTrackerEntry = "balance_sheet"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "quarterlyBalance":
                df['annualReport'] = 0
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                tickerTrackerEntry = "balance_sheet"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "annualCash":
                df['annualReport'] = 1
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                df.rename({"proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet":
                           "proceedsFromIssuanceOfLongTermDebt"}, axis=1, inplace=True)
                tickerTrackerEntry = "cash_flow"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "quarterlyCash":
                df['annualReport'] = 0
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                df.rename({"proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet":
                           "proceedsFromIssuanceOfLongTermDebt"}, axis=1, inplace=True)
                tickerTrackerEntry = "cash_flow"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "annualEarnings":
                continue
            
            elif typeFile == "quarterlyEarnings":
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                tickerTrackerEntry = "earnings"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "Overview":
                df = pd.DataFrame(df)
                df["Label"] = df.index
                df.rename(index={"Symbol":"ticker_symbol", \
                                 "52WeekHigh":"high_52week", \
                                 "52WeekLow":"low_52week", \
                                 "50DayMovingAverage":"moving_average_50d", \
                                 "200DayMovingAverage":"moving_average_200d"}, inplace=True)
                df = df.transpose()
                df.drop('Label', inplace = True)
                df['recordDate'] = df["LatestQuarter"]
                tickerTrackerEntry = "fund_overview"
                infoDate = df["LatestQuarter"]
                
            elif typeFile == "annualIncome":
                df['annualReport'] = 1
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                tickerTrackerEntry = "income_statement"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "quarterlyIncome":
                df['annualReport'] = 0
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                tickerTrackerEntry = "income_statement"
                infoDate = str(df.index.max().date())
            
            
            # Validates that the date associated with the most recent time the
            # data was current is actually formated as a date.
            try:
                self.validate(infoDate)
            except:
                infoDate = "2021-08-06"
            
            # Name of the table that the data needs to be entered into, as
            # determined by the file name and the converstion table above.
            tableName = conversion[typeFile]
            
            # Save the data into teh SQLite database.  Takes several minutes
            # for the series of files I had (~4.6 GB when complete)
            self.saveToSQL(df, tableName, ticker_symbol)
            
            # Update the ticker table data status; sets flags for whether there
            # were any errors downloading the data (no), whether the data in the
            # database is valid (yes), and when the data was last requested from 
            # the AlphaVantage API.  
            #
            # The 'histLengthString' variable is set in the "Time Series Daily"
            # if block above, or consists of a new line if the file is not the
            # daily time series.  The value set in the string represents the
            # number of daily records available for a given stock, allowing
            # a list of stocks with at least that many trading days available 
            # to be generated quickly.
            
            cmd_string = "UPDATE ticker_symbol_list \n" + \
                         "SET 'data_" + tickerTrackerEntry + "' = ?, \n" + \
                         "    'error_" + tickerTrackerEntry + "' = 0" + \
                         histLengthString + \
                         "WHERE ticker_symbol = ?;\n"
            argList = (infoDate, ticker_symbol)
            
            # Execute the SQLite transaction.
            cur.execute(cmd_string, argList)
        
        
        self.workingFileText = ""
    
    
    
    def validate(self, date_text):
        # Checks the string entered to see if it can be parsed into a date.
        try:
            datetime.datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")
    
    
    
    def loadTickerFromSQL(self, tableName, ticker_symbol = "*", propertyName = "*"):
        dat = self.DB.stockDB.cursor()
        
        queryString = "SELECT ? From ?\n"
        queryString += "WHERE ticker_symbol = ?;"
        
        argList = (propertyName, tableName, ticker_symbol)
        
        query = dat.execute(queryString, argList)
        cols = [column[0] for column in query.description]
        results = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
        return results
    
    
    def autoUpdate(self, metrics = None, olderThan = None, stockList = None, missing = True, error = False, limit = None):
        
        # metrics = None
        # olderThan = None
        # stockList = None
        # missing = True
        # error = False
        # limit = None
        
        cur = self.DB.stockDB.cursor()
        conversion = {"time":"daily_adjusted",
                      "balance":"balance_sheet",
                      "cash":"cash_flow",
                      "earnings":"earnings",
                      "overview":"fund_overview",
                      "income":"income_statement"}
        
        if metrics == None:
            metrics = ["time", "balance", "cash", "earnings", "overview", "income"]
        
        if olderThan != None:
            metricListString = "OR data_[METRICS] < " + str(datetime.date.today()-datetime.timedelta(days = olderThan))
        else:
            metricListString = ""
        
        
        columnString = ""
        metricString = ""
        queryStringBase  = "SELECT ticker_symbol, hist_length, recordDate" +\
                           "[METRIC_LIST] FROM ticker_symbol_list \n"
        queryStringBase += "WHERE NOT 1=1 " 
        
        if missing:
            metricListString += " OR data_[METRICS] IS NULL \n"
            metricListString += " OR error_[METRICS] = -1 \n"
        if error:
            metricListString += " OR error_[METRICS] = 1 \n"
        if stockList != None:
            metricListString += "AND ("
            for stock in stockList:
                metricListString += "ticker_symbol = " + stock + " OR "
            metricListString = metricListString[:-4] + ") \n"
        
        for metric in metrics:
            columnString += ", data_[METRICS], error_[METRICS]"
            columnString = columnString.replace("[METRICS]", conversion[metric])
            metricString += metricListString.replace("[METRICS]", conversion[metric])
        
        
        queryString  = queryStringBase.replace("[METRIC_LIST]", columnString)
        queryString += metricString
        
        query = cur.execute(queryString)
        cols = [column[0] for column in query.description]
        results = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
        
        
        if limit == None:
            pass
        elif isinstance(limit, int):
            results = results.head(limit)
        else:
            return -1
                
        for index, stock in results.iterrows():
            for metric in metrics:
                if metric == "time":
                    self.getTimeSeriesDaily(stock["ticker_symbol"])
                elif metric == "balance":
                    self.getBalanceSheet(stock["ticker_symbol"])
                elif metric == "cash":
                    self.getCashFlow(stock["ticker_symbol"])
                elif metric == "earnings":
                    self.getEarnings(stock["ticker_symbol"])
                elif metric == "overview":
                    self.getFundamentalOverview(stock["ticker_symbol"])
                elif metric == "income":
                    self.getIncomeStatement(stock["ticker_symbol"])





class database:
    def __init__(self, databaseSaveFile = "SQLiteDB/stockData.db"):
        if(os.path.exists(databaseSaveFile)):
            self.stockDB = sqlite3.connect(databaseSaveFile)
        else:
            try:
                self.stockDB = self.createStockDatabase(databaseSaveFile)
                self.DBcursor = self.stockDB.cursor()
            except:
                print("Failed to connect to stock database.")
                sys.exit()
                
    
    def createStockDatabase(self, fileName):
        currentDir = os.getcwd()
        try:
            dirs = fileName.split("/")
            while('.' in dirs):
                dirs.remove('.')
            
            for i in range(len(dirs)-1):
                os.mkdir(dirs[i])
                os.chdir(dirs[i])
                
            os.chdir(currentDir)
            self.createSchema(fileName)
            
            return sqlite3.connect(fileName)
        
        except:
            print("Could not connect to database.")
            sys.exit()
    
    
    def createSchema(self, fileName):
        conn = sqlite3.connect(fileName)
        cur = conn.cursor()
        print("Opened database successfully.")
        
        
        cur.execute('''CREATE TABLE fund_overview (
                id                          INTEGER    PRIMARY KEY    AUTOINCREMENT,
                ticker_symbol               TEXT       NOT NULL,
                recordDate                  TEXT       NOT NULL,
                cik                         INTEGER    NOT NULL,
                name						TEXT       NOT NULL,
                assetType					TEXT,
                description					TEXT,
                exchange					TEXT,
                currency					TEXT,
                country						TEXT,
                sector						TEXT,
                industry					TEXT,
                address						TEXT,
                fiscalYearEnd				TEXT,
                latestQuarter				TEXT,
                marketCapitalization		INTEGER,
                ebitda						INTEGER,
                peRatio						REAL,
                pegRatio					REAL,
                bookValue					REAL,
                dividendPerShare			REAL,
                dividendYield				REAL,
                eps							REAL,
                revenuePerShareTTM			REAL,
                profitMargin				REAL,
                operatingMarginTTM			REAL,
                returnOnAssetsTTM			REAL,
                returnOnEquityTTM			REAL,
                revenueTTM					INTEGER,
                grossProfitTTM				INTEGER,
                dilutedEPSTTM				REAL,
                quarterlyEarningsGrowthYOY	REAL,
                quarterlyRevenueGrowthYOY	REAL,
                analystTargetPrice			REAL,
                trailingPE					REAL,
                forwardPE					REAL,
                priceToSalesRatioTTM		REAL,
                priceToBookRatio			REAL,
                evToRevenue					REAL,
                evToEBITDA					REAL,
                beta						REAL,
                high_52week					REAL,
                low_52week					REAL,
                moving_average_50d			REAL,
                moving_average_200d			REAL,
                sharesOutstanding			INTEGER,
                sharesFloat     			INTEGER,
                sharesShort    			    INTEGER,
                sharesShortPriorMonth	    INTEGER,
                shortRatio                  REAL,
                shortPercentOutstanding     REAL,
                shortPercentFloat           REAL,
                percentInsiders             REAL,
                percentInstitutions         REAL,
                forwardAnnualDividendRate   REAL,
                forwardAnnualDividendYield  REAL,
                payoutRatio                 REAL,
                lastSplitFactor             REAL,
                lastSplitDate               TEXT,
                dividendDate				TEXT,
                exDividendDate				TEXT,
                UNIQUE(recordDate, ticker_symbol) ON CONFLICT REPLACE );''')
        
        
                
        cur.execute('''CREATE TABLE daily_adjusted (
                id                  INTEGER     PRIMARY KEY      AUTOINCREMENT,
                ticker_symbol       TEXT        NOT NULL,
                recordDate          TEXT        NOT NULL,
                open                REAL,
                high                REAL,
                low                 REAL,
                close               REAL,
                adj_close           REAL,
                volume              REAL,
                dividend            REAL,
                split               REAL,
                UNIQUE(recordDate, ticker_symbol) ON CONFLICT REPLACE );''')

        
        
        cur.execute('''CREATE TABLE income_statement (
                id                                 INTEGER    PRIMARY KEY    AUTOINCREMENT,
                ticker_symbol                      TEXT       NOT NULL,
                recordDate                         TEXT       NOT NULL,
                annualReport                       INTEGER    NOT NULL,
                reportedCurrency                   TEXT,
                grossProfit                        REAL,
                totalRevenue                       INTEGER,
                costOfRevenue                      INTEGER,
                costofGoodsAndServicesSold         INTEGER,
                operatingIncome                    INTEGER,
                sellingGeneralAndAdministrative    INTEGER,
                researchAndDevelopment             INTEGER,
                operatingExpenses                  INTEGER,
                investmentIncomeNet                INTEGER,
                netInterestIncome                  INTEGER,
                interestIncome                     INTEGER,
                interestExpense                    INTEGER,
                nonInterestIncome                  INTEGER,
                otherNonOperatingIncome            INTEGER,
                depreciation                       INTEGER,
                depreciationAndAmortization        INTEGER,
                incomeBeforeTax                    INTEGER,
                incomeTaxExpense                   INTEGER,
                interestAndDebtExpense             INTEGER,
                netIncomeFromContinuingOperations  INTEGER,
                comprehensiveIncomeNetOfTax        INTEGER,
                ebit                               INTEGER,
                ebitda                             INTEGER,
                netIncome                          INTEGER,
                UNIQUE(recordDate, ticker_symbol, annualReport) ON CONFLICT REPLACE );''')
                 
        
        
        cur.execute('''CREATE TABLE  balance_sheet (
                id                                      INTEGER    PRIMARY KEY    AUTOINCREMENT,
                ticker_symbol                           TEXT       NOT NULL,
                recordDate      						TEXT       NOT NULL,
                annualReport                            INTEGER    NOT NULL,
                reportedCurrency						TEXT,
                totalAssets								INTEGER,
                totalCurrentAssets						INTEGER,
                cashAndCashEquivalentsAtCarryingValue	INTEGER,
                cashAndShortTermInvestments				INTEGER,
                inventory								INTEGER,
                currentNetReceivables					INTEGER,
                totalNonCurrentAssets					INTEGER,
                propertyPlantEquipment					INTEGER,
                accumulatedDepreciationAmortizationPPE	INTEGER,
                intangibleAssets						INTEGER,
                intangibleAssetsExcludingGoodwill		INTEGER,
                goodwill								INTEGER,
                investments								INTEGER,
                longTermInvestments						INTEGER,
                shortTermInvestments					INTEGER,
                otherCurrentAssets						INTEGER,
                otherNonCurrrentAssets					INTEGER,
                totalLiabilities						INTEGER,
                totalCurrentLiabilities					INTEGER,
                currentAccountsPayable					INTEGER,
                deferredRevenue							INTEGER,
                currentDebt								INTEGER,
                shortTermDebt							INTEGER,
                totalNonCurrentLiabilities				INTEGER,
                capitalLeaseObligations					INTEGER,
                longTermDebt							INTEGER,
                currentLongTermDebt						INTEGER,
                longTermDebtNoncurrent					INTEGER,
                shortLongTermDebtTotal					INTEGER,
                otherCurrentLiabilities					INTEGER,
                otherNonCurrentLiabilities				INTEGER,
                totalShareholderEquity					INTEGER,
                treasuryStock							INTEGER,
                retainedEarnings						INTEGER,
                commonStock								INTEGER,
                commonStockSharesOutstanding			INTEGER,
                UNIQUE(recordDate, ticker_symbol, annualReport) ON CONFLICT REPLACE );''')



        cur.execute('''CREATE TABLE  earnings (
                id                      INTEGER    PRIMARY KEY    AUTOINCREMENT,
                ticker_symbol           TEXT       NOT NULL,
                recordDate      		TEXT       NOT NULL,
                reportedDate			TEXT,
                reportedEPS				INTEGER,
                estimatedEPS			INTEGER,
                surprise				INTEGER,
                surprisePercentage		INTEGER,
                UNIQUE(recordDate, ticker_symbol) ON CONFLICT REPLACE );''')



        cur.execute('''CREATE TABLE  cash_flow (
                id                                      INTEGER    PRIMARY KEY    AUTOINCREMENT,
                ticker_symbol                           TEXT       NOT NULL,
                recordDate      						TEXT       NOT NULL,
                annualReport                            INTEGER    NOT NULL,
                reportedCurrency						TEXT,
                operatingCashflow						INTEGER,
                paymentsForOperatingActivities			INTEGER,
                proceedsFromOperatingActivities			INTEGER,
                changeInOperatingLiabilities			INTEGER,
                changeInOperatingAssets					INTEGER,
                depreciationDepletionAndAmortization	INTEGER,
                capitalExpenditures						INTEGER,
                changeInReceivables						INTEGER,
                changeInInventory						INTEGER,
                profitLoss								INTEGER,
                cashflowFromInvestment					INTEGER,
                cashflowFromFinancing					INTEGER,
                proceedsFromRepaymentsOfShortTermDebt	INTEGER,
                paymentsForRepurchaseOfCommonStock		INTEGER,
                paymentsForRepurchaseOfEquity			INTEGER,
                paymentsForRepurchaseOfPreferredStock	INTEGER,
                dividendPayout							INTEGER,
                dividendPayoutCommonStock				INTEGER,
                dividendPayoutPreferredStock			INTEGER,
                proceedsFromIssuanceOfCommonStock		INTEGER,
                proceedsFromIssuanceOfLongTermDebt		INTEGER,
                proceedsFromIssuanceOfPreferredStock	INTEGER,
                proceedsFromRepurchaseOfEquity			INTEGER,
                proceedsFromSaleOfTreasuryStock			INTEGER,
                changeInCashAndCashEquivalents			INTEGER,
                changeInExchangeRate					INTEGER,
                netIncome								INTEGER,
                UNIQUE(recordDate, ticker_symbol, annualReport) ON CONFLICT REPLACE );''')
        
        
        
        cur.execute('''CREATE TABLE  errors_tracker (
                id                                      INTEGER    PRIMARY KEY    AUTOINCREMENT,
                ticker_symbol                           TEXT       NOT NULL,
                recordTime                              REAL       NOT NULL,
                recordDate      						TEXT,
                errorType      						    TEXT,
                errorSource      						TEXT,
                errorMessage      						TEXT,
                UNIQUE(recordTime, ticker_symbol) ON CONFLICT FAIL );''')
        
        
        
        cur.execute('''CREATE TABLE ticker_symbol_list (
                id                          INTEGER    PRIMARY KEY    AUTOINCREMENT,
                ticker_symbol               TEXT       NOT NULL,
                recordDate      			TEXT       NOT NULL,
                name                        TEXT,
                exchange                    TEXT,
                
                hist_length                 INTEGER    DEFAULT -1      NOT NULL,
                error_fund_overview         INTEGER    DEFAULT -1      NOT NULL,
                error_income_statement      INTEGER    DEFAULT -1      NOT NULL,
                error_earnings              INTEGER    DEFAULT -1      NOT NULL,
                error_daily_adjusted        INTEGER    DEFAULT -1      NOT NULL,
                error_balance_sheet         INTEGER    DEFAULT -1      NOT NULL,
                error_cash_flow             INTEGER    DEFAULT -1      NOT NULL,
                
                data_fund_overview          TEXT,
                data_income_statement       TEXT,
                data_earnings               TEXT,
                data_daily_adjusted         TEXT,
                data_balance_sheet          TEXT,
                data_cash_flow              TEXT,
                
                UNIQUE(ticker_symbol) ON CONFLICT IGNORE );''')
        
        
        
        print("Schema created successfully.")
        
        conn.close()



if __name__ == "__main__":
    t_start = time.time()
    
    info = getData()
    cur = info.DB.stockDB.cursor()
    
    
    # df_0 = info.loadStockListCSV("TotalStockList.csv", True)
    # cur = info.DB.stockDB.cursor()
    # cur.execute("SELECT * FROM ticker_symbol_list")
    # rows_0 = cur.fetchall()
    
    # count = 0
    # for i in rows_0:
    #     count += 1
    # print("Rows in Ticker List:  " + str(count))
    
    # info.copyPickleToSQL("./Stocks/")
    info.autoUpdate()
    
    t_1 = time.time()
    
    
    # info.loadStockListCSV("TotalStockList.csv", True)
    # info.copyPickleToSQL("./Stocks/")
    
    # cur.execute("SELECT * FROM ticker_symbol_list")
    # tickers = cur.fetchall()
    
    # cur.execute("SELECT * FROM daily_adjusted")
    # daily = cur.fetchall()
    
    # cur.execute("SELECT * FROM balance_sheet")
    # balance = cur.fetchall()
    
    # cur.execute("SELECT * FROM cash_flow")
    # cash = cur.fetchall()
    
    # cur.execute("SELECT * FROM earnings")
    # earnings = cur.fetchall()
    
    # cur.execute("SELECT * FROM fund_overview")
    # overview = cur.fetchall()
    
    # cur.execute("SELECT * FROM income_statement")
    # income = cur.fetchall()
    
    
    t_2 = time.time()
    
    
    # df_1 = info.getCashFlow('TSLA')
    # cur.execute("SELECT * FROM cash_flow")
    # rows_1 = cur.fetchall()
    
    # count = 0
    # for i in rows_1:
    #     count += 1
    # print("Rows in Cash Statement table:  " + str(count))
    

    # df_2 = info.getTimeSeriesDaily('TSLA')
    # cur = info.DB.stockDB.cursor()
    # cur.execute("SELECT * FROM daily_adjusted")
    # rows_2 = cur.fetchall()
    
    # count = 0
    # for i in rows_2:
    #     count += 1
    # print("Rows in Daily Price table:  " + str(count))


    # df_3 = info.getFundamentalOverview('TSLA')
    # cur = info.DB.stockDB.cursor()
    # cur.execute("SELECT * FROM fund_overview")
    # rows_3 = cur.fetchall()
    
    # count = 0
    # for i in rows_3:
    #     count += 1
    # print("Rows in Fundamental table:  " + str(count))


    # df_4 = info.getEarnings('TSLA')
    # cur = info.DB.stockDB.cursor()
    # cur.execute("SELECT * FROM earnings")
    # rows_4 = cur.fetchall()
    
    # count = 0
    # for i in rows_4:
    #     count += 1
    # print("Rows in Earnings table:  " + str(count))
    
    
    # df_5 = info.getIncomeStatement('TSLA')
    # cur = info.DB.stockDB.cursor()
    # cur.execute("SELECT * FROM income_statement")
    # rows_5 = cur.fetchall()
    
    # count = 0
    # for i in rows_5:
    #     count += 1
    # print("Rows in Income table:  " + str(count))
    
    
    
    # df_6 = info.getBalanceSheet('TSLA')
    # cur = info.DB.stockDB.cursor()
    # cur.execute("SELECT * FROM balance_sheet")
    # rows_6 = cur.fetchall()
    
    # count = 0
    # for i in rows_6:
    #     count += 1
    # print("Rows in Balance Sheet table:  " + str(count))
    
    
    
    # info.getBalanceSheet('AFTM')
    # info.getTimeSeriesDaily('AFTM')
    # info.getCashFlow('AFTM')
    # info.getFundamentalOverview('AFTM')
    # info.getIncomeStatement('AFTM')
    # info.getEarnings('AFTM')
    
    
    
    # cur = info.DB.stockDB.cursor()
    # cur.execute("SELECT * FROM errors_tracker")
    # rows_e = cur.fetchall()
    
    # print("Rows in Ticker List:  " + str(len(rows_e)))
    
    
    # query = cur.execute("SELECT * FROM ticker_symbol_list")
    # cols = [column[0] for column in query.description]
    # results = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
    
    
    t_end = time.time()


    # alphaVantageBaseURL = "https://www.alphavantage.co/query?"
    # API_KEY = "NYLUQRD89OVSIL3I"
    # ticker_symbol = "TSLA"
    # requestURL = alphaVantageBaseURL + "function=TIME_SERIES_DAILY_ADJUSTED&" + \
    #                   "outputsize=full&symbol=" + ticker_symbol + "&apikey=" + API_KEY
    
    # for i in range(1):
    #     response = requests.get(requestURL)
    #     data = response.json()

    
    # info = getData()
    # cur = info.DB.stockDB.cursor()
    
    # query = cur.execute("SELECT * FROM ticker_symbol_list WHERE error_fund_overview = -1")
    # cols = [column[0] for column in query.description]
    # results = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
    
    # for index, stock in results.iterrows():
    #     info.getFundamentalOverview(stock["ticker_symbol"])









