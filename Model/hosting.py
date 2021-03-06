#!flask/bin/python
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import flash
from flask import request
from flask import redirect
from flask import url_for

from flask_wtf import FlaskForm, Form
import capstoneModels

from wtforms import StringField
from wtforms import SubmitField
from wtforms import RadioField
from wtforms import SelectField
from wtforms import SelectMultipleField
from wtforms import widgets

from wtforms.validators import InputRequired
import os
import datetime as dt
import pandas as pd

app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY




indicatorList = [("0",  'MA20'), 
                 ("1",  'MA50'), 
                 ("2",  'MACD12'), 
                 ("3",  'MACD19'), 
                 ("4",  'OBV'), 
                 ("5",  'RSI'), 
                 ("6",  'BOLLINGER20'), 
                 ("7",  'BOLLINGER50'), 
                 ("8",  'IDEAL')]

extraList = [("00",  'OPEN'), 
             ("01",  'HIGH'), 
             ("02",  'LOW'), 
             ("03",  'CLOSE'), 
             ("04",  'ADJCLOSE'), 
             ("05",  'ADJRATIO'), 
             ("06",  'VOLUME'), 
             ("07",  'VOL20'), 
             ("08",  'VOL50'),
             ("09",  'SPLIT'), 
             ("10",  'DIVIDEND'), 
             ("11",  'TOTALCHANGE'),
             ("12",  'DAYCHANGE'),
             ("13",  'TP20'), 
             ("14",  'TP50'), 
             ("15",  'IDEAL_TRIG'), 
             ("16",  'IDEAL_LOW'), 
             ("17",  'IDEAL_HIGH')]




class flaskFunctions:
    def __init__(self, mod):
        self.mod = mod
        self.mod.LSTM_load()
    
    # Create A Search Form
    class SearchForm(FlaskForm):
        radio        = RadioField(label  = "radioButton", 
                                  choices=[("True",  "Company Name"), 
                                           ("False", "Ticker Symbol")], 
                                  default="False")
        searched     = StringField(label = "searched",    validators=[InputRequired()])
        submitSearch = SubmitField(label = "submitSearch")
        
        
        
    class tickerSubmitForm(FlaskForm):
        symbol       = StringField(label = "symbol",    validators=[InputRequired()])
        submitTicker = SubmitField(label = "submitTicker")
    
    
    
    class checkBoxForm(Form):
        symbol       = StringField(label = "symbol",    validators=[InputRequired()])
        submitTicker = SubmitField(label = "submitTicker")
        
        extras       = SelectMultipleField(label = "extras",
                                        widget = widgets.TableWidget(),
                                        option_widget = widgets.CheckboxInput(),
                                        choices = extraList)
        
        indicator = SelectMultipleField(label = "indicators", 
                                        widget = widgets.TableWidget(),
                                        option_widget = widgets.CheckboxInput(),
                                        choices = indicatorList)
    
    
    
    
    def clearSearch(self):
        f = open("./templates/tickerSearchResults.html", "w")
        f.write("")
        f.close()
        
        
        
    def clearStockPriceData(self):
        f = open("./templates/stockPriceData.html", "w")
        f.write("")
        f.close()
    
    
    
    def clearStockPediction(self):
        f = open("./templates/stockPrediction.html", "w")
        f.write("")
        f.close()
    
    
    
    def clearUpdate(self):
        f = open("./templates/stockUpdate.html", "w")
        f.write("")
        f.close()
        
    
    
    
    def searchSubmital(self, form):
        f = open("./templates/tickerSearchResults.html", "w")
        
        if form.validate_on_submit():
            
            searchString = form.searched.data
            companyName  = form.radio.data
            companyName  = True if companyName == "True" else False
            returnedData = self.mod.analysis.searchForStock(searchString.upper(), 
                                                            name = companyName)
            
            f.write(returnedData)
        
        f.close()
    
    
    
        
    def symbolPriceSubmital(self, form):
        f = open("./templates/stockPriceData.html", "w")
        if form.validate_on_submit():
            
            indicators = []
            extras = []
            
            for key in form.indicator.data:
                k = int(key.replace("\"",""))
                indicators.append(indicatorList[k][1])
            
            for key in form.extras.data:
                k = int(key.replace("\"",""))
                extras.append(extraList[k][1])
            
            # block needed because the function that gets this information expects
            # at least one indicator to be supplied
            removeData = {}
            
            if indicators == []:
                indicators = ["MA20"]
                removeData["MA20"] = True
                
            for col in ["OPEN", "CLOSE", "ADJCLOSE"]:
                if col not in extras:
                    removeData[col] = True
                else:
                    extras.remove(col)
            
            
            searchString = form.symbol.data
            
            returnedData, t = self.mod.analysis.loadFromDB(tickerList   = [searchString.upper()], 
                                                           indicators   = indicators,
                                                           extras       = extras,
                                                           withTriggers = False)
            
            
            inv_map = {v: k for k, v in self.mod.analysis._dailyConversionTable.items()}
            inv_map["ticker_symbol"] = "SYMBOL"
            inv_map["recordDate"] = "DATE"
            colNames = list(returnedData.columns)
            renameDict = {}
            
            for col in colNames:
                renameDict[col] = inv_map[col]
            
            returnedData.rename(columns = renameDict, inplace = True)
            
            returnedData.set_index("DATE")
            
            # remove the columns that were returned but not requested (tied to underlying function)
            for col in removeData.keys():
                returnedData.drop(labels = col, inplace=True, axis=1)
            
            f.write(returnedData.to_html(classes = "tickertable\" id=\"companyList"))
        
        f.close()
        
        
        
        
    def predictLSTM(self, form, function):
        self.clearStockPediction()
        f = open("./templates/stockPrediction.html", "w")
        if form.validate_on_submit():
            
            searchString = form.symbol.data
            timestampstr = str(dt.datetime.now()).replace(":", "").replace(" ","").replace("-","").replace(".","")
            
            prediction, evaluation = self.mod.LSTM_eval(ticker = searchString.upper(), 
                                                        evaluate = False, 
                                                        predict = True,
                                                        savePlt = True, 
                                                        plotWindow = 600,
                                                        predLen = 15,
                                                        look_back = 120)
            
            stringVar = ""
            stringVar += "\n<br>\n<img src=\"static\LSTM_1_" + timestampstr + ".png\" width=\"700\">"
            stringVar += "\n<br><br>\n<img src=\"static\LSTM_2_" + timestampstr + ".png\" width=\"700\">\n<pre>"
            for key in evaluation.keys():
                stringVar += str(key) + ":  " + str(evaluation[key]) + "<br>"
            stringVar += "<form method=\"get\" action=\"static\LSTM_prediction.csv\">\n" 
            stringVar += "<button type=\"submit\">Download CSV</button>\n"
            stringVar += "</form>"
            
            f.write(stringVar)
        
        f.close()
    
    
    
    def predictARIMA(self, form):
        f = open("./templates/stockPrediction.html", "w")
        if form.validate_on_submit():
            
            searchString = form.symbol.data
            timestampstr = str(dt.datetime.now()).replace(":", "").replace(" ","").replace("-","").replace(".","")
            
            model_autoARIMA, fitted, endData, mets = self.mod.autoARIMA(ticker = searchString.upper(), 
                                                                        evaluate = False, 
                                                                        savePlt = True,
                                                                        timestampstr = timestampstr)
            
            stringVar = ""
            stringVar += "\n<br>\n<img src=\"static\ARIMA_1_" + timestampstr + ".png\" width=\"700\">"
            stringVar += "\n<br><br>\n<img src=\"static\ARIMA_2_" + timestampstr + ".png\" width=\"700\">\n<pre>"
            for key in mets.keys():
                stringVar += str(key) + ":  " + str(mets[key]) + "<br>"
            stringVar += "<br>" + str(fitted.summary()) + "</pre>"
            stringVar += "<form method=\"get\" action=\"static\ARIMA_prediction.csv\">\n" 
            stringVar += "<button type=\"submit\">Download CSV</button>\n"
            stringVar += "</form>"
            
            f.write(stringVar)
            
            endData.to_csv(path_or_buf = "./static/ARIMA_prediction.csv")
        
        f.close()
        return
    
    
    
    
    def predictTree(self, form):
    
        f = open("./templates/stockPrediction.html", "w")
        if form.validate_on_submit():
            
            searchString = form.symbol.data
            timestampstr = str(dt.datetime.now()).replace(":", "").replace(" ","").replace("-","").replace(".","")
            
            tree, endData, mets = self.mod.Trees(ticker = searchString.upper(), 
                                                     evaluate = False, 
                                                     savePlt = True,
                                                     timestampstr = timestampstr)
             
            stringVar = ""
            stringVar += "\n<br>\n<img src=\"static\Tree_1_" + timestampstr + ".png\" width=\"700\">\n<pre>"
            for key in mets.keys():
                stringVar += str(key) + ":  " + str(mets[key]) + "<br>"
            stringVar += "<br></pre>"
            stringVar += "<form method=\"get\" action=\"static\Tree_prediction.csv\">\n" 
            stringVar += "<button type=\"submit\">Download CSV</button>\n"
            stringVar += "</form>"
            
            f.write(stringVar)
            
            endData.to_csv(path_or_buf = "./static/Tree_prediction.csv")
        
        f.close()
        return
    
    
    
    def predictLinear(self, form):
    
        f = open("./templates/stockPrediction.html", "w")
        if form.validate_on_submit():
            
            searchString = form.symbol.data
            timestampstr = str(dt.datetime.now()).replace(":", "").replace(" ","").replace("-","").replace(".","")
            
            Linear, endData, mets = self.mod.linearRegression(ticker = searchString.upper(), 
                                                              evaluate = False, 
                                                              savePlt = True,
                                                              timestampstr = timestampstr)
             
            stringVar = ""
            stringVar += "\n<br>\n<img src=\"static\Linear_1_" + timestampstr + ".png\" width=\"700\">\n<pre>"
            for key in mets.keys():
                stringVar += str(key) + ":  " + str(mets[key]) + "<br>"
            stringVar += "<br></pre>"
            stringVar += "<form method=\"get\" action=\"static\Linear_prediction.csv\">\n" 
            stringVar += "<button type=\"submit\">Download CSV</button>\n"
            stringVar += "</form>"
            
            f.write(stringVar)
            
            endData.to_csv(path_or_buf = "./static/Linear_prediction.csv")
        
        f.close()
        return
    
    
        
    
    def symbolUpdate(self, form):
        if form.validate_on_submit():
            
            searchString = form.symbol.data
            
            self.mod.stockdata.autoUpdate(stockList = [searchString.upper()])
            self.mod.indicators.autoSaveIndicators()
            self.mod.indicators.populateSummaryData()
            self.mod.analysis.storeTriggers(tickerList = self.mod.analysis._tickerList)
            
    
    
    
    
mod = capstoneModels.MLmodels(dataBaseSaveFile = "./stockData.db", 
                              dataBaseThreadCheck = False,
                              splitDate = "2020-01-01")

flaskFunc = flaskFunctions(mod)



@app.route('/', methods=["GET", "POST"])
def index():
    return render_template("./index.html")


@app.route('/lstm', methods=["GET", "POST"])
def lstm():
    searchForm = flaskFunc.SearchForm()
    predictForm = flaskFunc.tickerSubmitForm()
    
    flaskFunc.clearSearch()
    flaskFunc.searchSubmital(form = searchForm)
    flaskFunc.predictLSTM(form = predictForm, function = "LSTM")

    return render_template("./lstm.html",
                            searchForm  = searchForm,
                            predictForm = predictForm)



@app.route('/arima', methods=["GET", "POST"])
def arima():
    searchForm  = flaskFunc.SearchForm()
    predictForm = flaskFunc.tickerSubmitForm()
    
    flaskFunc.clearSearch()
    flaskFunc.searchSubmital(form = searchForm)
    flaskFunc.predictARIMA(form = predictForm)

    return render_template("./arima.html",
                            searchForm  = searchForm,
                            predictForm = predictForm)




@app.route('/tree', methods=["GET", "POST"])
def tree():
    searchForm  = flaskFunc.SearchForm()
    predictForm = flaskFunc.tickerSubmitForm()
    
    flaskFunc.clearSearch()
    flaskFunc.searchSubmital(form = searchForm)
    flaskFunc.predictTree(form = predictForm)

    return render_template("./tree.html",
                            searchForm  = searchForm,
                            predictForm = predictForm)




@app.route('/linear', methods=["GET", "POST"])
def linear():
    searchForm  = flaskFunc.SearchForm()
    predictForm = flaskFunc.tickerSubmitForm()
    
    flaskFunc.clearSearch()
    flaskFunc.searchSubmital(form = searchForm)
    flaskFunc.predictLinear(form = predictForm)

    return render_template("./linear.html",
                            searchForm  = searchForm,
                            predictForm = predictForm)




@app.route('/update', methods=["GET", "POST"])
def update():
    form1 = flaskFunc.SearchForm()
    form2 = flaskFunc.tickerSubmitForm()
    
    flaskFunc.clearSearch()
    flaskFunc.searchSubmital(form = form1)

    return render_template("./update.html",
                           searchForm = form1,
                           dataForm   = form2)




@app.route('/raw', methods=["GET", "POST"])
def raw():
    searchForm = flaskFunc.SearchForm()
    checkBoxForm = flaskFunc.checkBoxForm()
    
    flaskFunc.searchSubmital(form = searchForm)
    flaskFunc.symbolPriceSubmital(form = checkBoxForm)
    
    
    selectExtras = checkBoxForm.extras.data
    selectIndicator = checkBoxForm.indicator.data
    return render_template("./raw.html",
                            searchForm = searchForm,
                            rawForm = checkBoxForm)



@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
    




@app.route('/query', methods=["GET"])
def query():
    
    if(request.method == 'GET'):
        function  = request.args.get('function')
        ticker    = request.args.get('symbol')
        startDate = request.args.get('date')
        
        timestampstr = str(dt.datetime.now()).replace(":", "").replace(" ","").replace("-","").replace(".","")
        
        
        if function == "LSTM":
            prediction, evaluation, testX, testY = \
                flaskFunc.mod.LSTM_eval(ticker = ticker.upper(), 
                                        evaluate = False, 
                                        savePlt = True, 
                                        plotWindow = 600,
                                        predLen = 15,
                                        look_back = 120)
            
            df = pd.DataFrame(prediction, 
                              columns = ['open',
                                         'high',
                                         'low', 
                                         'close',
                                         'vol'])
            return df.to_json()
            
            
        elif function == "ARIMA":
            model, fitted, endData, mets = \
                flaskFunc.mod.autoARIMA(ticker = ticker.upper(), 
                                        evaluate = False, 
                                        savePlt = True,
                                        timestampstr = timestampstr)
            
            endData = endData.drop(columns = ["adj_close"])
            endData = endData.dropna(how='any', axis=0)
            return endData.to_json()
        
            
        elif function == "Tree":
            model, endData, mets = flaskFunc.mod.Trees(ticker = ticker.upper(), 
                                                       evaluate = False, 
                                                       savePlt = True)
            
            endData = endData["treePrediction"]
            endData = endData.dropna(how='any', axis=0)
            return endData.to_json()
            
        elif function == "Linear":
            model, endData, mets = flaskFunc.mod.linearRegression(ticker = ticker.upper(), 
                                                                  evaluate = False, 
                                                                  savePlt = True)
            
            endData = endData["val"]
            endData = endData.dropna(how='any', axis=0)
            return endData.to_json()
            
        elif function == "update":
            flaskFunc.mod.stockdata.autoUpdate(stockList = [ticker.upper()])
            flaskFunc.mod.indicators.autoSaveIndicators()
            flaskFunc.mod.indicators.populateSummaryData()
            flaskFunc.mod.analysis.storeTriggers(tickerList = flaskFunc.mod.analysis._tickerList)
            
        elif function == "raw":
            
            indicators = []
            extras = []
            
            for i in indicatorList:
                indicators.append(i[1])
            for e in extraList:
                extras.append(e[1])
            for e in ["OPEN", "CLOSE", "ADJCLOSE"]:
                extras.remove(e)
                
            
            returnedData, t = flaskFunc.mod.analysis.loadFromDB(tickerList   = [ticker.upper()], 
                                                                indicators   = indicators,
                                                                extras       = extras)
            inv_map = {v: k for k, v in flaskFunc.mod.analysis._dailyConversionTable.items()}
            inv_map["ticker_symbol"] = "SYMBOL"
            inv_map["recordDate"] = "DATE"
            colNames = list(returnedData.columns)
            renameDict = {}
            
            for col in colNames:
                renameDict[col] = inv_map[col]
            
            returnedData.rename(columns = renameDict, inplace = True)
            
            returnedData.set_index("DATE")
            
            return returnedData.to_json()
    
    
    return render_template("./index.html")





if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    