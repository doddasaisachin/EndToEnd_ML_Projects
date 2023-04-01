from flask import Flask,request,render_template,jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

ridge_model=pickle.load(open('model/ridge.pkl','rb'))
scaler=pickle.load(open('model/scaler.pkl','rb'))

application = Flask(__name__)
app=application

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def predict_data():
    if request.method=='GET':
        return render_template('home.html')
    else:
        temperature=float(request.form.get('Temperature'))
        rh=float(request.form.get('RH'))
        ws=float(request.form.get('Ws'))
        rain=float(request.form.get('Rain'))
        ffmc=float(request.form.get('FFMC'))
        dmc=float(request.form.get('DMC'))
        isi=float(request.form.get('ISI'))
        classes=float(request.form.get('Classes'))
        region=float(request.form.get('Region'))

        new_data_scaled=scaler.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])

if __name__=='__main__':
    app.run(host='0.0.0.0')