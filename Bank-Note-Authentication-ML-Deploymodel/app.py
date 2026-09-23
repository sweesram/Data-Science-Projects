import uvicorn
from fastapi import FastAPI
from Banknote import BankNote
import numpy as np
import pickle
import pandas as pd

app=FastAPI()
pickle_in=open("classifier.pkl","rb") #model ai lod pannum pothu rbla load pannanum write pannum poth wb
classifier=pickle.load(pickle_in)

@app.post('/predict')
def predict_banknote(data:BankNote):
    data=data.dict()
    varience=data['varience']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entrophy=data['entrophy']
    # print(classifier.predict([[varience,skewness,curtosis,entrophy]]))
    prediction=classifier.predict([[varience,skewness,curtosis,entrophy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return{
        'prediction':prediction
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5001)