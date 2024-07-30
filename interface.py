from flask import Flask,request,render_template,jsonify
import pandas as pd
import numpy as np
from utils import Heart_Disease


app = Flask(__name__)
HD = Heart_Disease()

@app.route("/",methods = ["GET","POST"])
def Final_Prediction():
    data = request.form

    age        = float(data["age"])
    sex        = float(data["sex"])
    cp         = float(data["cp"])
    trestbps   = float(data["trestbps"])
    chol       = float(data["chol"])
    fbs        = float(data["fbs"])
    restecg    = float(data["restecg"])
    thalach    = float(data["thalach"])
    exang      = float(data["exang"])
    oldpeak    = float(data["oldpeak"])
    slope      = float(data["slope"])
    ca         = float(data["ca"])
    thal       = float(data["thal"])

    # data = np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])

    predict = HD.Predictive_model(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)

    print([predict])
    return [predict]

if __name__ =="__main__":
    app.run()


    

