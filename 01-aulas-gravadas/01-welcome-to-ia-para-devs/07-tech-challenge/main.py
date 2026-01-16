import pickle
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
modelo_salvo = pickle.load(open("pipe_rf_model.pkl", "rb"))

class HealthData(BaseModel):
    age: int
    sbp: float
    hba1c: float
    bmi: float
    gender: float
    high_bp: float
    chf: float
    smoking: float


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(data: HealthData):
    input_data = [[
        data.age,
        data.sbp,
        data.hba1c,
        data.bmi,
        data.gender,
        data.high_bp,
        data.chf,
        data.smoking
    ]]
    prediction = modelo_salvo.predict([input_data])
    return {"prediction": prediction[0], "input": data}