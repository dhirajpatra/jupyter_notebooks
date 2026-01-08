from flask import Flask

import pandas as pd
import numpy as np
from sklearn.external import joblib
from sklearn.preprocessing import StandardScaler
from flask import request, jsonify

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('../boston_housing_prediction_model.joblib')
scaler = joblib.load('../boston_housing_prediction_scaler.joblib')

# Define the route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json['data']
    
    # Convert the input data to a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Scale the input data
    scaled_data = scaler.transform(df)
    
    # Make the prediction
    prediction = model.predict(scaled_data)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
