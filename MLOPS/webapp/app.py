from flask import Flask

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from flask import request, jsonify, render_template
import os

app = Flask(__name__)

# Load the pre-trained model and scaler
# Use absolute path to avoid FileNotFoundError
model_path = os.path.join(os.path.dirname(__file__), '../boston_housing_prediction.joblib')

# List of all 13 features expected by the model in order to make predictions for boston housing dataset
FEATURE_NAMES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

@app.route('/')
def home():
    return render_template('index.html')

# Function to scale the input data
def scale(payload):
    # Initialize the scaler
    scaler = StandardScaler().fit(payload)
    
    # Transform the payload
    return scaler.transform(payload)

# Define the route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    """
    Input sample:
    {
        "CHAS": { "0": 0 }, "RM": { "0": 6.575 },
        "TAX": { "0": 296 }, "PTRATIO": { "0": 15.3 },
        "B": { "0": 396.9 }, "LSTAT": { "0": 4.98 },
        "AGE": { "0": 15.2 }, "DIS": { "0": 4.0900 },
        "RAD": { "0": 5 }, "ZN": { "0": 0 },
        "NOX": { "0": 0.538 }, "INDUS": { "0": 11.9300 },
        "CRIM": { "0": 0.00632 }
    }
    Output sample:
    { "prediction": [ 20.35373177134412 ] }
    """
    # 1. Get the json data
    # Load the pre-trained model and scaler
    clf = joblib.load(model_path)

    # Get the input data from the request
    inference_payload = pd.DataFrame(request.json)

    # 2. Fill missing features with 0 (or training means) to reach 13 features
    for col in FEATURE_NAMES:
        if col not in inference_payload.columns:
            inference_payload[col] = 0  
    
    # 3. Ensure the columns are in the exact same order as training
    inference_payload = inference_payload[FEATURE_NAMES]
    
    # 4. Scale the input data
    inference_payload = scale(inference_payload)
    
    # Make the prediction
    prediction = list(clf.predict(inference_payload))
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
