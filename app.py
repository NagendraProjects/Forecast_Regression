# coding: utf-8

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
#from sklearn import metrics
from flask import Flask, request, render_template
#import re
#import math
import pickle
#from xgboost import XGBRegressor

app = Flask("__name__")

d1=pd.read_excel('Weekly data Refinance Volumes_upto_aug1.xlsx')
future_data  = d1[d1['Refinance'].isna()]
base_data = d1[~d1['Refinance'].isna()]
x=base_data[['Year','Mortgage Rate','Inflation']]
y=base_data['Refinance']

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=24)
model = RandomForestRegressor(n_estimators=500,criterion='mae',max_features='auto', max_depth=2,min_samples_split=2,min_samples_leaf=4,bootstrap=True,n_jobs=-1, random_state=102)
model.fit(X_train, y_train)
print("Train Accuracy:",model.score(X_train, y_train))
print("Test Accuracy:",model.score(X_test, y_test))

@app.route("/")
def loadPage():
	return render_template('index.html', query="")



@app.route("/predict", methods=['POST'])
def predict():    
    #model = pickle.load(open("RF_tuned.sav", "rb"))
    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']    
       
    
    data = [[inputQuery1, inputQuery2, inputQuery3]]
    
    # Create the pandas DataFrame 
    new_df = pd.DataFrame(data, columns = ['Year', 'Mortgage Rate', 'Inflation'])
    prediction = model.predict(new_df)
    o1 = "The forecasting value is {}".format(prediction)
    
    return render_template('home.html', output1=o1, query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'])
    
if __name__ == "__main__":
    app.run(debug=True)
