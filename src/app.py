from utils import db_connect
engine = db_connect()

# your code here
import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Define base path explicitly
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load model and scaler from base path
model = load_model(os.path.join(BASE_DIR, 'src/flight_status_model.h5'))
scaler = joblib.load(os.path.join(BASE_DIR, 'src/scaler.save'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form['airport_country_code']),
                float(request.form['airport_continent']),
                float(request.form['arrival_airport']),
                float(request.form['week_day']),
                float(request.form['day']),
                float(request.form['month'])]

    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    predicted_class = np.argmax(prediction, axis=1)[0]

    flight_mapping = {0: 'Cancelled', 1: 'Delayed', 2: 'On-time'}
    result = flight_mapping[predicted_class]

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
