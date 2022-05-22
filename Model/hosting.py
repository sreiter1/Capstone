# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:10:33 2022

@author: sreit
"""

from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            input_values = request.json
            input_df = pd.json_normalize(input_values) 
            prediction = lr.predict(input_df)
            return jsonify({'prediction': str(prediction)})


        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)
    
    
    
# {"id":{"0":1},"budget":{"0":14000000},"popularity":{"0":6.575393},"runtime":{"0":93.0}}
# sample input through postman