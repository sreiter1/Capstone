#!flask/bin/python
from flask import Flask, jsonify, render_template, flash, request, redirect, url_for, session
from flask_wtf import FlaskForm, Form
import model
from wtforms import StringField, SubmitField, RadioField, SelectField, SelectMultipleField, widgets
from wtforms.validators import InputRequired
import os

app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY




class flaskFunctions:
    def __init__(self, mod):
        self.mod = mod
    
    # Create A Search Form
    class SearchForm(FlaskForm):
        radio        = RadioField(label  = "radioButton", choices=[(True,  "Company Name"), 
                                                               (False, "Ticker Symbol")])
        searched     = StringField(label = "searched",    validators=[InputRequired()])
        submitSearch = SubmitField(label = "submitSearch")
        
        
        
    class tickerSubmitForm(FlaskForm):
        symbol       = StringField(label = "symbol",    validators=[InputRequired()])
        submitTicker = SubmitField(label = "submitTicker")
    
    
    
    class addlRawFeatures(FlaskForm):
        indicator = SelectField(label = "indicators", choices=[("0",  'MA20'), 
                                                               ("1",  'MA50'), 
                                                               ("2",  'MACD12'), 
                                                               ("3",  'MACD19'), 
                                                               ("4",  'OBV'), 
                                                               ("5",  'RSI'), 
                                                               ("6",  'BOLLINGER20'), 
                                                               ("7",  'BOLLINGER50'), 
                                                               ("8",  'IDEAL')])
        
        extras    = SelectField(label = "extras",     choices=[("0",  'DAYCHANGE'), 
                                                               ("1",  'ADJCLOSE'), 
                                                               ("2",  'IDEAL_LOW'), 
                                                               ("3",  'HIGH'), 
                                                               ("4",  'CLOSE'), 
                                                               ("5",  'OPEN'), 
                                                               ("6",  'VOL20'), 
                                                               ("7",  'SPLIT'), 
                                                               ("8",  'LOW'), 
                                                               ("9",  'IDEAL_TRIG'), 
                                                               ("10", 'DIVIDEND'), 
                                                               ("11", 'TOTALCHANGE'), 
                                                               ("12", 'TP50'), 
                                                               ("13", 'TP20'), 
                                                               ("14", 'ADJRATIO'), 
                                                               ("15", 'VOLUME'), 
                                                               ("16", 'IDEAL_HIGH'), 
                                                               ("17", 'VOL50')])
        
        symbol    = StringField(label = "symbol",     validators=[InputRequired()])
        submitRaw = SubmitField(label = "submitRaw")
        
        
        
    
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
            returnedData = self.mod.analysis.searchForStock(searchString, name = companyName)
            
            f.write(returnedData)
        
        f.close()
        
        
        
    def symbolPriceSubmital(self, form):
        f = open("./templates/stockPriceData.html", "w")
        if form.validate_on_submit():
            
            searchString = form.symbol.data
            returnedData, t = self.mod.analysis.loadFromDB(tickerList = [searchString.upper()], 
                                                           indicators = ["MA20", "MA50"],
                                                           extras     = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"],
                                                           withTriggers = False)
            
            f.write(returnedData.to_html(classes = "tickertable\" id=\"companyList"))
        
        f.close()
        
        
        
    def symbolPredict(self, form, function, date = -1):
        f = open("./templates/stockPrediction.html", "w")
        if form.validate_on_submit():
            pass
            # searchString = form.symbol.data
            # returnedData, t = self.mod.analysis.loadFromDB(tickerList = [searchString.upper()], 
            #                                                indicators = ["MA20", "MA50"],
            #                                                withTriggers = False)
            
            # f.write(returnedData.to_html(classes = "tickertable\" id=\"companyList"))
        
        f.close()
        
        
    
    def symbolUpdate(self, form):
        f = open("./templates/stockUpdate.html", "w")
        if form.validate_on_submit():
            pass
            # searchString = form.symbol.data
            # returnedData, t = self.mod.analysis.loadFromDB(tickerList = [searchString.upper()], 
            #                                                indicators = ["MA20", "MA50"],
            #                                                withTriggers = False)
            
            
            # f.write(returnedData.to_html(classes = "tickertable\" id=\"companyList"))
        
        f.close()
        


class ChoiceObj(object):
    def __init__(self, name, choices):
        # this is needed so that BaseForm.process will accept the object for the named form,
        # and eventually it will end up in SelectMultipleField.process_data and get assigned
        # to .data
        setattr(self, name, choices)

class MultiCheckboxField(SelectMultipleField):
    widget = widgets.TableWidget()
    option_widget = widgets.CheckboxInput()

    # uncomment to see how the process call passes through this object
    # def process_data(self, value):
    #     return super(MultiCheckboxField, self).process_data(value)

class ColorLookupForm(Form):
    submit = SubmitField('Save')
    colors = MultiCheckboxField(None)

allColors = ( 'red', 'pink', 'blue', 'green', 'yellow', 'purple' )


        


mod = model.MLmodels(dataBaseSaveFile = "./stockData.db", 
                     dataBaseThreadCheck = False,
                     splitDate = "2020-01-01")


indicList = [(i,e) for i,e in zip(range(len(mod.indicatorList.keys())), 
                                            mod.indicatorList.keys())]

set1 = set(mod._dailyConversionTable.keys())
set2 = set(mod.indicatorList.keys())
extraList = set1 - set2
extraList = [(i,e) for i,e in zip(range(len(extraList)), extraList)]


flaskFunc = flaskFunctions(mod)
        


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template("./index.html")


@app.route('/lstm', methods=["GET", "POST"])
def lstm():
    form1 = flaskFunc.SearchForm()
    form2 = flaskFunc.tickerSubmitForm()
    
    flaskFunc.clearSearch()
    flaskFunc.searchSubmital(form = form1)
    dataBack = flaskFunc.symbolPredict(form = form2)

    return render_template("./lstm.html",
                            searchForm = form1,
                            dataForm   = form2)



@app.route('/arima', methods=["GET", "POST"])
def arima():
    form1 = flaskFunc.SearchForm()
    form2 = flaskFunc.tickerSubmitForm()
    
    flaskFunc.clearSearch()
    flaskFunc.searchSubmital(form = form1)

    return render_template("./arima.html",
                            searchForm = form1,
                            dataForm   = form2)




@app.route('/tree', methods=["GET", "POST"])
def tree():
    form1 = flaskFunc.SearchForm()
    form2 = flaskFunc.tickerSubmitForm()
    
    flaskFunc.clearSearch()
    flaskFunc.searchSubmital(form = form1)

    return render_template("./tree.html",
                           searchForm = form1,
                           dataForm   = form2)




@app.route('/linear', methods=["GET", "POST"])
def linear():
    form1 = flaskFunc.SearchForm()
    form2 = flaskFunc.tickerSubmitForm()
    
    flaskFunc.clearSearch()
    flaskFunc.searchSubmital(form = form1)

    return render_template("./linear.html",
                           searchForm = form1,
                           dataForm   = form2)




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
# def color():
#     selectedChoices = ChoiceObj('colors', session.get('selected') )
#     form = ColorLookupForm(obj=selectedChoices)
#     form.colors.choices =  [(c, c) for c in allColors]

#     if form.validate_on_submit():
#         session['selected'] = form.colors.data
#         return redirect(url_for('.color'))
#     else:
#         print(form.errors)
#     return render_template("./raw.html",
#                            trialForm = form,
#                            rawForm = form)


def raw():
    form1 = flaskFunc.SearchForm()
    form2 = flaskFunc.addlRawFeatures()
    form2.extras
    
    flaskFunc.clearSearch()
    flaskFunc.searchSubmital(form = form1)
    dataBack = flaskFunc.symbolPriceSubmital(form = form2)

    return render_template("./raw.html",
                            searchForm = form1,
                            rawForm    = form2)




@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
    




if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    