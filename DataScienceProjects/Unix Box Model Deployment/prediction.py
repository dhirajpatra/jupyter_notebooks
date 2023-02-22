import numpy as np
import pandas as pd
import itertools
import sys
import joblib 
from sklearn.metrics import accuracy_score, confusion_matrix

print("Data is reading")
BaseDir = sys.argv[1]
inputpath = BaseDir+"/DataInput/"
outputpath = BaseDir+"/PredictionOutPut/"

#Reading the data
mydf=pd.read_csv(inputpath+"/TestData.csv")

#Do Preprocessing
print("Preprocessing in progress")
predictors = mydf['text']
y_test = mydf.label

tfidfVectorizer = joblib.load(open(BaseDir+"/TrainingOutPut/TFidfVecotorizer08252021.sav","rb"))
tfidfTest=tfidfVectorizer.transform(predictors)

#Read Model
print("Model is loading.............")
Model = joblib.load(open(BaseDir+"/TrainingOutPut/Model08252021.sav","rb"))

print("Prediction in progress.............")
#prediction
y_pred=Model.predict(tfidfTest)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: ' + str(score))

#printing confusion matrix
print(confusion_matrix(y_test,y_pred, labels=['FAKE','REAL']))

mydf["predicted_column"] = y_pred

mydf.to_csv(BaseDir+"/PredictionOutPut/Prediction_Output.csv",index=False)

print("Prediction output written to disk successfully....")