import pandas as pd
import numpy as np
import pickle

class Heart_Disease():
    def pickle_files(self):
        with open("std_data.pkl","rb") as f:
            self.std_scaling = pickle.load(f)

        with open("knn_model.pkl","rb") as g:
            self.model = pickle.load(g)

    def Predictive_model(self,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
        self.pickle_files()
        test_array = np.zeros(13)
        print(test_array)
        
        test_array[0] = age
        test_array[1] = sex
        test_array[2] = cp
        test_array[3] = trestbps
        test_array[4] = chol
        test_array[5] = fbs
        test_array[6] = restecg
        test_array[7] = thalach
        test_array[8] = exang
        test_array[9] = oldpeak
        test_array[10] = slope
        test_array[11] = ca
        test_array[12] = thal

        print(test_array)

        scaled_array = self.std_scaling.transform([test_array])

        prediction = self.model.predict(scaled_array)

        print("Prediction :",prediction[0])

        if prediction[0] == 0:
            print("Patient Not Suffer with Heart Disease")
            return "Patient Not Suffer with Heart Disease"

        else :
            print("Patient Suffer with Heart Disease")
            return "Patient Suffer with Heart Disease"