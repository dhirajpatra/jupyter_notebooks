#This is my demo python script for parametrization
print("This is python demo for parametrization")

import sys
import pandas as pd
MyData = pd.read_csv("electronic-card-transactions-july-2021-csv-tables.csv")

print("Data Set has " +str(MyData.shape[0] )+" rows and "  +str(MyData.shape[1] )+" columns")

#Calcualte the Dollar volume for Year 2007  - Non Parametric way
#Temp_Data = MyData.loc[MyData["Year"]==2003]
#print("Without Parameter execution " + str(sum(Temp_Data["Data_value"])))



#Parametric way of deploying code
year_value = sys.argv[1]


Temp_Data = MyData.loc[MyData["Year"]==int(year_value)]
print("With Parameter execution "+str(sum(Temp_Data["Data_value"])))

#Taking one more parameter as input

NoOfParams = len(sys.argv)-1
if (NoOfParams==2):
    month_value = sys.argv[2]
    Temp_Data = MyData.loc[(MyData["Year"]==int(year_value) )&  (MyData["Month"]==int(month_value) )]
    print("With Parameter execution "+str(sum(Temp_Data["Data_value"])))




