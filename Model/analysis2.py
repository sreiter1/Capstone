# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:00:28 2021

@author: sreit
"""

import pandas as pd
import numpy as np
import sqlite3
import commonUtilities
import matplotlib.pyplot as plt

#import statsmodels.api as sm


import warnings
warnings.filterwarnings("ignore")


class missingTicker(Exception):
    pass



class analysis:
    def __init__(self, dataBaseSaveFile = "./stockData.db"):
        self.DB = sqlite3.connect(dataBaseSaveFile)
        self._cur = self.DB.cursor()
        self._tickerList = []   # Empty list that gets filled with a list of tickers to be considered
        self.tradingDateSet = []  # List of dates in YYYY-MM-DD format that are trading dates in the database
        self.dailyTableNames = ["alpha", "yahoo"]
        
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
                              "BOLLINGER20": "bolinger_20",
                              "BOLLINGER50": "bolinger_50"}
    
    
    
    def compareDataSources(self, ticker):
        self.validate.validateString(ticker)
        
        df = []
        for i in range(len(self.dailyTableNames)):
            table = self.dailyTableNames[i]
            queryString  = "SELECT recordDate, adj_close, volume "
            queryString += "FROM " + table + "_daily \n"
            queryString += "WHERE ticker_symbol = ?\n"
            
            query = self._cur.execute(queryString, [ticker])
            cols = [column[0] for column in query.description]
            df.append(pd.DataFrame.from_records(data = query.fetchall(), columns = cols))
            if df[i]["recordDate"].empty:
                raise missingTicker("Ticker   " + str(ticker).rjust(6) + "   Missing from table   " + str(table) + "_daily.")
        
        
        columnNames = ["missing_dates", "entries", "startDate", 
                       "abs_delta_adj_close", "abs_delta_volume", 
                       "percent_delta_close", "percent_delta_volume"]
        stats = pd.DataFrame(columns = columnNames, index = self.dailyTableNames)
        
        for i in range(len(self.dailyTableNames)):
            table = self.dailyTableNames[i]
            totalDates = [n for n in self.tradingDateSet if n >= df[i]["recordDate"].min()]
            totalDates = set(totalDates)
            tickerDates = set(df[i]["recordDate"])
            missingDates = list(totalDates - tickerDates)
            missingDates.sort()
            stats.at[table, "missing_dates"] = missingDates
            
            stats.at[table, "entries"] = df[i]["recordDate"].count()
            stats.at[table, "startDate"] = df[i]["recordDate"].min()
            
            if i == 0:
                stats.at[table, "abs_delta_adj_close"] = 0
                stats.at[table, "abs_delta_volume"] = 0
                stats.at[table, "percent_delta_close"] = 0
                stats.at[table, "percent_delta_volume"] = 0
            else:
                stats.at[table, "abs_delta_adj_close"] = [abs(a - b) for a, b in zip(df[0]["adj_close"], df[i]["adj_close"])]
                stats.at[table, "abs_delta_volume"] = [abs(a - b) for a, b in zip(df[0]["volume"], df[i]["volume"])]
                stats.at[table, "percent_delta_close"] = [100*abs(a - b) / a for a, b in zip(df[0]["adj_close"], df[i]["adj_close"])]
                stats.at[table, "percent_delta_volume"] = [100*abs(a - b) / a for a, b in zip(df[0]["volume"], df[i]["volume"])]
        
        return stats
    
    
    
    def fillTradingDates(self):
        
        for table in self.dailyTableNames:
            print("\rCollecting dates from table:  " + table + "                      ", end = "")
            queryString  = "SELECT DISTINCT recordDate "
            queryString += "FROM " + table + "_daily \n"
            
            query = self._cur.execute(queryString)
            cols = [column[0] for column in query.description]
            tickerList = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
            tickerList = list(tickerList["recordDate"])
            self.tradingDateSet += tickerList
        
        print("\rSorting Data...                                                ", end = "")
        self.tradingDateSet = set(self.tradingDateSet)
        self.tradingDateSet = list(self.tradingDateSet)
        self.tradingDateSet.sort()
        saveArray = [[x] for x in self.tradingDateSet]
        
        print("\rSaving Data...                                                ", end = "")
        
        query  = "INSERT OR IGNORE INTO trading_dates (trading_date)\n"
        query += "VALUES(?)"
        self._cur.executemany(query, saveArray)
        self.DB.commit()
        
        return
    
    
    
    def filterStocksFromDataBase(self, 
                                 dailyLength = 0,            # time = 1250 is 5 years of trading days.  Default is keep all.
                                 marketCap = 0,              # commonStockSharesOutstanding from balance * price from daily
                                 sector = None,              # fundamental overview
                                 exchange = None,            # fundamental overview
                                 country = None,             # fundamental overview
                                 industry = None,            # fundamental overview
                                 peRatio = 0.0,              # EPS under earnings, price under daily
                                 profitMargin = 0.0,         # profit/loss in cash flow; net income / total revenue --> income statement
                                 shareHolderEquity = 0,      # balance sheet
                                 EPS = 0.0,                  # from earnings
                                 maxDailyChange = None,      # maximum daily % change
                                 minDailyChange = None,      # minimum daily % change
                                 minDailyVolume = None       # minimum daily volume; 
                                 ):     
                       
        # Creates a list of stocks that meet the requirements passed to the function.
        
        # check inputs
        self.validate.validateInteger(dailyLength)
        self.validate.validateInteger(marketCap)
        self.validate.validateInteger(shareHolderEquity)
        
        self.validate.validateNum(peRatio)
        self.validate.validateNum(profitMargin)
        self.validate.validateNum(EPS)
        if maxDailyChange != None:
            self.validate.validateNum(maxDailyChange)
        if minDailyChange != None:
            self.validate.validateNum(minDailyChange)
        if minDailyVolume != None:
            self.validate.validateNum(minDailyVolume)
        
        
        if sector is None: pass
        else: self.validate.validateListString(sector)
        
        if exchange is None: pass
        else: self.validate.validateListString(exchange)
        
        if country is None: pass
        else: self.validate.validateListString(country)
        
        if industry is None: pass
        else: self.validate.validateListString(industry)
        
        
        # start SQL query string.  "WHERE 1=1 AND ..." allows for the addition 
        # of other requirements within the request string later
        query  = "SELECT [COLUMNS]"
        query += "FROM summary_data \n"
        query += "WHERE 1=1 [CONDITIONS]"
                
        # create strings that will be inserted into the query string later, as
        # well as an empty list for the arguments that will also be passed
        columnString = "ticker_symbol, \n  "
        conditionString = "\n  AND "
        argList = []
        
        # add requirements for the shortest price history length that will be allowed
        if dailyLength != 0:
            argList.append(str(dailyLength))
            columnString += "daily_length, \n  "
            conditionString += "daily_length >= ? \n  AND "
            
        # add requirements for the minimum market capitalization that will be allowed
        if marketCap != 0:
            argList.append(str(marketCap))
            columnString += "market_capitalization, \n  "
            conditionString += "market_capitalization >= ? \n  AND "
            
        # add requirements for the minimum PE ratio that will be allowed
        if peRatio != 0:
            argList.append(str(peRatio))
            columnString += "pe_ratio, \n  "
            conditionString += "pe_ratio >= ? \n  AND "
            
        # add requirements for the minimum profit margin that will be allowed
        if profitMargin != 0:
            argList.append(str(profitMargin))
            columnString += "profit_margin, \n  "
            conditionString += "profit_margin >= ? \n  AND "
            
        # add requirements for the minimum shareholder equity that will be allowed
        if shareHolderEquity != 0:
            argList.append(str(shareHolderEquity))
            columnString += "share_holder_equity, \n  "
            conditionString += "share_holder_equity >= ? \n  AND "
            
        # add requirements for the minimum earnings per share that will be allowed
        if EPS != 0:
            argList.append(str(EPS))
            columnString += "earnings_per_share, \n  "
            conditionString += "earnings_per_share >= ? \n  AND "
        
        # add requirements for the maximum daily change that will be allowed
        if maxDailyChange != None:
            maxDailyChange = abs(maxDailyChange)
            argList.append(str(maxDailyChange))
            columnString += "max_daily_change, \n  "
            conditionString += "max_daily_change <= ? \n  AND "
        
        # add requirements for the maximum daily change that will be allowed
        if minDailyChange != None:
            minDailyChange = -abs(minDailyChange)
            argList.append(str(minDailyChange))
            columnString += "min_daily_change, \n  "
            conditionString += "min_daily_change >= ? \n  AND "
        
        # add requirements for the maximum daily change that will be allowed
        if minDailyVolume != None:
            minDailyVolume = abs(minDailyVolume)
            argList.append(str(minDailyVolume))
            columnString += "min_daily_volume, \n  "
            conditionString += "min_daily_volume >= ? \n  AND "
        
        
        # add requirements for the sectors that will be allowed
        if sector is not None:
            conditionString += " ("
            for sect in sector:
                argList.append(str(sect))
                columnString += "sector, \n  "
                conditionString += "sector = ? \n OR "
            
            conditionString = conditionString[:-5] + " ) \n  AND "
            
        
        # add requirements for the exchanges that will be allowed
        if exchange is not None:
            conditionString += " ("
            for ex in exchange:
                argList.append(str(ex))
                columnString += "exchange, \n  "
                conditionString += "exchange = ? \n OR "
            
            conditionString = conditionString[:-5] + " ) \n  AND "
            
        
        # add requirements for the exchanges that will be allowed
        if country is not None:
            conditionString += " ("
            for co in country:
                argList.append(str(co))
                columnString += "country, \n  "
                conditionString += "country = ? \n OR "
            
            conditionString = conditionString[:-5] + " ) \n  AND "
            
        
        # add requirements for the exchanges that will be allowed
        if industry is not None:
            conditionString += " ("
            for ind in industry:
                argList.append(str(ind))
                columnString += "industry, \n  "
                conditionString += "industry = ? \n OR "
            
            conditionString = conditionString[:-5] + " ) \n  AND "
            
        
        # remove extra characters at the end of the string to format it correctly
        # for the SQL query
        columnString = columnString[:-5] + " \n"
        conditionString = conditionString[:-7] + "; \n"
        
        # replace the placeholders in the original query string with specific 
        # values
        query = query.replace("[COLUMNS]", columnString)
        query = query.replace("[CONDITIONS]", conditionString)
        
        # execute the SQL query and format the response as a pandas dataframe
        result = self._cur.execute(query, argList)
        cols = [column[0] for column in result.description]
        DF_tickerList = pd.DataFrame.from_records(data = result.fetchall(), columns = cols)
        
        # update the internal list of tickers to match those that are in the response
        self._tickerList = list(DF_tickerList["ticker_symbol"])
        
        if self._tickerList == []:
            warnings.warn("tickerList is an empty list.  Is the table 'summary_data' empty?  Run filter fuction with option updateBeforeFilter = True")
        
        # return the dataframe
        return DF_tickerList
    
    
    
    def listUnique(self, extended = False):
        # returns a dictionary of the unique values available in the summary_data table
        
        # start SQL query string.  "WHERE 1=1 AND ..." allows for the addition 
        # of other requirements within the request string later
        query  = "SELECT [COLUMNS]"
        query += "FROM summary_data \n"
                
        # create strings that will be inserted into the query string later, as
        # well as an empty list for the arguments that will also be passed
        columnString = "ticker_symbol, \n  "
        
        # populate the columns to be gathered
        if extended == True:
            columnString += "date_calculated, \n  "
            columnString += "daily_length, \n  "
            columnString += "earnings_per_share, \n  "
            columnString += "profit_margin, \n  "
            columnString += "share_holder_equity, \n  "
            columnString += "common_shares_outstanding, \n  "
            columnString += "current_price, \n  "
            columnString += "market_capitalization, \n  "
            columnString += "pe_ratio, \n  "
            columnString += "avg_return, \n  "
            columnString += "std_return, \n  "
            columnString += "comp_return, \n  "
            columnString += "comp_stddev, \n  "
            columnString += "max_daily_change, \n  "
            columnString += "min_daily_change, \n  "
            columnString += "min_daily_volume, \n  "
        
        
        columnString += "country, \n  "
        columnString += "exchange, \n  "
        columnString += "sector, \n  "
        columnString += "industry \n  "
        
        # execute the SQL query and format the response as a pandas dataframe
        query = query.replace("[COLUMNS]", columnString)
        result = self._cur.execute(query)
        cols = [column[0] for column in result.description]
        df = pd.DataFrame.from_records(data = result.fetchall(), columns = cols)
        
        self.uniqueValues = {}
        self.uniqueValueCounts = {}
        for colName in df:
            self.uniqueValues[colName] = df[colName].unique()
            countList = []
            for countTuple in df[colName].value_counts().iteritems():
                countList.append(countTuple)
            self.uniqueValueCounts[colName] = countList
            
        return df
    
    
    
    def storeTriggers(self, OBVshift = 5, rsiLower = 30, rsiUpper = 70, indicators = [], tickerList = []):
        
        if indicators == []:
            indicators = self.indicatorList.keys()
        if tickerList == []:
            tickerList = self._tickerList
            
        for ind in indicators:
            assert ind in self.indicatorList.keys(), "\nError, 'indicators' passed to 'storeTriggers()' not in 'indicatorList' key list."
        
        for tick in tickerList:
            print("\rProcessing ticker:  " + str(tick).rjust(6) + ".                                       ", end = "")
            
            queryString = "SELECT * " +\
                          "FROM daily_adjusted \n" +\
                          "WHERE ticker_symbol = ? "
            
            argList = [tick]
            
            # execute the SQL query and convert the response to a pandas dataframe
            query = self._cur.execute(queryString, argList)
            cols = [column[0] for column in query.description]
            rslt_df = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
            
            # for ind in self.indicatorList.values():
            #     triggerColumn = self.indicatorList[ind] + "_trig"
            #     rslt_df[triggerColumn] = None
            
            
            # start SQL query line for saving the triggers back to the database
            sqlString  = "UPDATE daily_adjusted \n SET "
            sqlArray = pd.DataFrame()
        
            for ind in self.indicatorList.values():
                triggerColumn = ind + "_trig"
                
                if "macd" in ind:
                    rslt_df[triggerColumn] = np.sign(rslt_df[ind])
                if "mvng_avg" in ind:
                    rslt_df[triggerColumn] = np.sign(rslt_df["adj_close"] - rslt_df[ind])
                if "bal_vol" in ind:
                    rslt_df[triggerColumn] = np.sign(rslt_df[ind] - rslt_df[ind].shift(periods = OBVshift, fill_value = 0))
                if "rsi" in ind:
                    temp = [1 if (x<rsiLower and y>rsiLower) else -1 if (x>rsiUpper and y<rsiUpper)
                            else 0 for x,y in zip(rslt_df[ind][:-1],rslt_df[ind][1:])]
                    temp.insert(0,0)
                    rslt_df[triggerColumn] = temp
                    rslt_df[triggerColumn] = rslt_df[triggerColumn].replace(0, method = "ffill")
                                
                # add each indicator trigger in the list of indicators to the 
                # SQL query, along with the associated values.
                sqlString += str(triggerColumn) + " = ?, \n     "
                sqlArray[triggerColumn] = rslt_df[triggerColumn]
                
            # finish the string, execute the SQL transaction, and commit the changes
            sqlString  = sqlString[:-8] + "\n"
            sqlString += "WHERE ticker_symbol = ? AND recordDate = ?; \n"
            sqlArray["ticker_symbol"] = tick
            sqlArray["recordDate"] = rslt_df["recordDate"]
            sqlArray = sqlArray.values.tolist()
            
            self._cur.executemany(sqlString, sqlArray)
            self.DB.commit()
    
    
    
    def plotIndicators(self, tickerList = [], indicators = []):
        print("\nProcessing plotting...")
        if indicators == []:
            indicators = self.indicatorList.keys()
        if tickerList == []:
            tickerList = self._tickerList
        
        loadedData, trigList = self.loadFromDB(tickerList = tickerList, indicators = indicators)
        
        for ind in indicators:
            indColName = self.indicatorList[ind]
            
            x_p = []
            x_n = []
            y_p = []
            y_n = []
            
            
            triggerLoc = list(np.where(np.diff(loadedData[indColName + "_trig"]))[0])
            
            for i in range(len(triggerLoc)-1):
                if loadedData[indColName + "_trig"][triggerLoc[i]] > 0:
                    y_p.append(list(loadedData["tradePrice"][triggerLoc[i]+1:triggerLoc[i+1]]/loadedData["tradePrice"][triggerLoc[i]]))
                    x_p.append(list(range(triggerLoc[i+1] - triggerLoc[i] - 1)))
                else:
                    y_n.append(list(loadedData["tradePrice"][triggerLoc[i]+1:triggerLoc[i+1]]/loadedData["tradePrice"][triggerLoc[i]]))
                    x_n.append(list(range(triggerLoc[i+1] - triggerLoc[i] - 1)))
            
            
            x_p = pd.Series([x for seg in x_p for x in seg])
            x_n = pd.Series([x for seg in x_n for x in seg])
            
            y_p = pd.Series([y/seg[0]-1 for seg in y_p for y in seg])
            y_n = pd.Series([y/seg[0]-1 for seg in y_n for y in seg])
            
            plt.figure()
            plt.scatter(x_p, y_p, marker = ".", s = 5, c = "#00cc00", label = "buy")
            plt.scatter(x_n, y_n, marker = ".", s = 5, c = "#ff0000", label = "sell")
            plt.title("Scatter Plot of time vs returns " + ind)
            plt.legend()
            
            
            avgNeg = y_n.mean()
            avgPos = y_p.mean()
            avgLen = len(loadedData)/len(triggerLoc)
            
            print("average " + ind + " positive return:   " + str(avgPos))
            print("average " + ind + " negative return:   " + str(avgNeg))
            print("average " + ind + " timeframe:         " + str(avgLen))
            print()
            
        
        
    
    
    def loadFromDB(self, tickerList = [], indicators = []):
        self.validate.validateListString(tickerList)
        self.validate.validateListString(indicators)
        for value in indicators:
            if value not in self.indicatorList.keys():
                raise ValueError("Indicators passed are not listed in analysis module.")
        
        results = pd.DataFrame()
        trigList = []
        
        for tick in tickerList:
            print("\rRetrieving ticker:  " + str(tick).rjust(6) + "  and indicators:  " + str(indicators) + ".             ", end = "")
            argList = []
            argList.append(tick)
            queryString = "SELECT [INDICATORS] adj_close, ticker_symbol " +\
                          "FROM daily_adjusted \n" +\
                          "WHERE ticker_symbol = ?;"
            
            # append to the SQL query string and argument list each ticker from the 
            # function inputs
            indicatorString = ""
            for ind in indicators:
                indicatorString += self.indicatorList[ind] + ", "
            
            queryString = queryString.replace("[INDICATORS]", indicatorString)
            
            query = self._cur.execute(queryString, argList)
            cols = [column[0] for column in query.description]
            results = results.append(pd.DataFrame.from_records(data = query.fetchall(), columns = cols), ignore_index=True)
            
            trigList.append(len(results.index))
        
        trigList.pop()
        
        return results, trigList






if __name__ == "__main__":
    ana = analysis()
    ana.filterStocksFromDataBase(dailyLength = 1250, maxDailyChange = 100, minDailyChange = -80, minDailyVolume = 1000)
    print("Number of stocks selected:  " + str(len(ana._tickerList)) + ".             ")
    
    ana.storeTriggers(tickerList = ana._tickerList)
    ana.plotIndicators(tickerList = ana._tickerList)
    
    
    
    
    
    
    
    
    
    
    
    