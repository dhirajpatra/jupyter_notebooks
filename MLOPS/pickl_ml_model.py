# ==========================================
# 0.1 Ingest
# ==========================================
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import read_csv, set_option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.metrics import mean_squared_error
import joblib  # Updated from sklearn.externals.joblib (which is deprecated)

boston_housing = "https://raw.githubusercontent.com/noahgift/boston_housing_pickle/master/housing.csv"
names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
    'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
df = read_csv(boston_housing, delim_whitespace=True, names=names)

print("--- First 5 rows of the dataset ---")
print(df.head())

# ==========================================
# 0.2 EDA (Exploratory Data Analysis)
# ==========================================
# CHAS- Charles River dummy variable(1 if tract bounds river; 0 otherwise)
# RM- average number of rooms per dwelling
# TAX- full-value property-tax rate per $10,000
# PTRATIO- pupil-teacher ratio by town
# Bk is the proportion of blacks by town
# LSTAT- % lower status of the population
# MEDV- Median value of owner-occupied homes in $1000’s

prices = df['MEDV']
df = df.drop(['CRIM', 'ZN', 'INDUS', 'NOX', 'AGE', 'DIS', 'RAD'], axis=1)
features = df.drop('MEDV', axis=1)

print("\n--- Features and Target after dropping columns ---")
print(df.head())

# ==========================================
# 0.3 Modeling
# ==========================================
# Split Data
# Split-out validation dataset
array = df.values
X = array[:, 0:6]
Y = array[:, 6]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=validation_size, random_state=seed
)

print("\n--- Validation Samples ---")
for sample in list(X_validation)[0:2]:
    print(f"X_validation {sample}")

# Tune
# Test options and evaluation metric using Root Mean Square error method
num_folds = 10
seed = 7
RMS = 'neg_mean_squared_error'
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=np.array([50, 100, 150, 200, 250, 300, 350, 400]))
model = GradientBoostingRegressor(random_state=seed)

# Note: Added shuffle=True to avoid warning/error in newer scikit-learn versions
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print(f"\nBest: {grid_result.best_score_} using {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"{mean} ({stdev}) with: {param}")

# 0.3.1 Fit Model
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)

# transform the validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print("\nMean Squared Error:\n")
print(mean_squared_error(Y_validation, predictions))

# 0.3.2 Evaluate
predictions = predictions.astype(int)
evaluate = pd.DataFrame({
    "Org House Price": Y_validation,
    "Pred House Price": predictions
})
evaluate["difference"] = evaluate["Org House Price"] - evaluate["Pred House Price"]
print("\n--- Evaluation Head ---")
print(evaluate.head())

print("\n--- Evaluation Describe ---")
print(evaluate.describe())

# ==========================================
# 0.4 Adhoc Predict
# ==========================================
actual_sample = df.head(1)
print("\n--- Actual Sample ---")
print(actual_sample)

adhoc_predict = actual_sample[["CHAS", "RM", "TAX", "PTRATIO", "B", "LSTAT"]]
print("\n--- Adhoc Predict Features ---")
print(adhoc_predict.head())

json_payload = adhoc_predict.to_json()
print("\n--- JSON Payload ---")
print(json_payload)

scaler = StandardScaler().fit(adhoc_predict)
scaled_adhoc_predict = scaler.transform(adhoc_predict)
print("\n--- Scaled Adhoc Predict ---")
print(scaled_adhoc_predict)

print("\n--- Model Prediction ---")
print(list(model.predict(scaled_adhoc_predict)))

# 0.4.1 Pickle the model
joblib.dump(model, 'boston_housing_prediction.joblib')
print("\nModel saved to 'boston_housing_prediction.joblib'")

# 0.4.2 Unpickle and predict
clf = joblib.load('boston_housing_prediction.joblib')

actual_sample2 = df.head(5)
print("\n--- Actual Sample 2 ---")
print(actual_sample2)

# Note: The original notebook used `actual_sample` here instead of `actual_sample2`
adhoc_predict2 = actual_sample[["CHAS", "RM", "TAX", "PTRATIO", "B", "LSTAT"]]
print("\n--- Adhoc Predict 2 Features ---")
print(adhoc_predict2.head())

# 0.4.3 scale input
scaler = StandardScaler().fit(adhoc_predict2)
scaled_adhoc_predict2 = scaler.transform(adhoc_predict2)
print("\n--- Scaled Adhoc Predict 2 ---")
print(scaled_adhoc_predict2)

# Use pickle loaded model
print("\n--- Loaded Model Prediction ---")
print(list(clf.predict(scaled_adhoc_predict2)))