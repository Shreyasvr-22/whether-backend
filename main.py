
from fastapi import FastAPI
from tensorflow.keras.models import load_model
import numpy as np
from pydantic import BaseModel
from typing import List

app = FastAPI()
model = load_model('rain_model.h5')  # Loads architecture + weights automatically [web:39]

class WeatherInput(BaseModel):
    temp: float
    humidity: float
    pressure: float
    wind_speed: float
    # Add your exact features from training

@app.post("/predict")
def predict_rain(data: WeatherInput):
    # Reshape to match your model's input shape (e.g., [1, timesteps, features])
    input_data = np.array([[data.temp, data.humidity, data.pressure, data.wind_speed]])
    input_data = input_data.reshape(1, 1, input_data.shape[1])  # For LSTM single step
    prediction = model.predict(input_data)[0][0]  # Rain probability %
    return {"rain_probability": float(prediction * 100)}
