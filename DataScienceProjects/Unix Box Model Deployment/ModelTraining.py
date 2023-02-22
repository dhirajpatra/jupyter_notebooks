import numpy as np
import pandas as pd
import itertools
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime,date,time
import joblib
now = datetime.now()
DateOfExec  = now.strftime('%m%d%Y')

print("Data Reading..........")
BaseDir = sys.argv[1]
inputpath = BaseDir+"/DataInput/"
outputpath = BaseDir+"/TrainingOutPut/"

#Reading the data
mydf=pd.read_csv(inputpath+"/fake_or_real_news.csv")

#print records
print("Separating training and test data.........")
print(mydf.head())
NoOfRows = mydf.shape[0]

TrainData = mydf[0:int(NoOfRows*0.8)]
TestData = mydf[int(NoOfRows*0.8):NoOfRows]

TestData.to_csv("DataInput/TestData.csv")

print("Data preprocessing..........")

#Splitting in test and train
x_train,x_test,y_train,y_test=train_test_split(TrainData['text'], TrainData.label, test_size=0.1, random_state=5)

#Initialiazing TfidfVectorizer
tfidfVectorizer=TfidfVectorizer(stop_words='english', max_df=0.8)

#creating tfidf vector for train and test
tfidfTtrain=tfidfVectorizer.fit_transform(x_train) 

print("Model Training..........")
#Initialize a RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(tfidfTtrain,y_train)


#write tfidf and model to disk
with open(outputpath+"TFidfVecotorizer"+DateOfExec+".sav","w") as fp:
    pass
joblib.dump(tfidfVectorizer, open(outputpath+"TFidfVecotorizer"+DateOfExec+".sav","wb"))

with open(outputpath+"Model"+DateOfExec+".sav","w") as fp:
    pass
joblib.dump(rfc, open(outputpath+"Model"+DateOfExec+".sav","wb"))


print("Model is trained and written to disk")
