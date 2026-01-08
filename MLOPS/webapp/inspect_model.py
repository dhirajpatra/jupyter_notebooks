import joblib
import os
import pandas as pd
import numpy as np

try:
    model_path = '../boston_housing_prediction.joblib'
    print(f"Loading from: {model_path}")
    clf = joblib.load(model_path)
    
    # Dummy data matching the shape from app.py
    # "CHAS": { "0": 0 }, "RM": { "0": 6.575 }, "TAX": { "0": 296 }, "PTRATIO": { "0": 15.3 }, "B": { "0": 396.9 }, "LSTAT": { "0": 4.98 }
    
    data = {
        "CHAS": [0],
        "RM": [6.575],
        "TAX": [296],
        "PTRATIO": [15.3],
        "B": [396.9],
        "LSTAT": [4.98]
    }
    df = pd.DataFrame(data)
    print("Input data:")
    print(df)
    
    # Try prediction without scaling
    prediction = clf.predict(df)
    print(f"Prediction (unscaled): {prediction}")

except Exception as e:
    print(f"Error: {e}")
