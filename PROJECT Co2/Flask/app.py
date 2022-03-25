# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:41:10 2020

@author: Adminr
"""
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle


app=Flask(__name__)
filepath="Model.pkl"
model=pickle.load(open(filepath,'rb'))

@app.route('/')
def home():
    return render_template("home.html")
@app.route('/Prediction',methods=['POST','GET'])
def prediction():# route which will take you to the prediction page
    return render_template('indexnew.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict',methods=["POST","GET"])# route to show the predictions in a web UI

def predict():
    input_feature=[float(x) for x in request.form.values() ]
    
    features_values=[np.array(input_feature)]
    
    feature_name=['Make', 'Vehicle_Class', 'Engine_Size', 'Cylinders',
       'Transmission', 'Fuel_Type', 'Fuel_Consumption_City',
       'Fuel_Consumption_Hwy', 'Fuel_Consumption_Comb(mpg)']
    
    x=pd.DataFrame(features_values,columns=feature_name)
    
    prediction=model.predict(x)
    
    print("Prediction is:",prediction)
    
    return render_template("resultnew.html",prediction=prediction[0])


if __name__=="__main__":
    app.run(debug=False)
    