#IMPORTS

import math
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


import csv


avg_year_built=1968
median_floor_count=3
building_dict={}

#Instantiate the dictionary of building data
i=0
while(i<=1448):
    building_dict[i]=[[],[],[],[]]
    i=i+1

with open('train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    #Skip the first line
    next(readCSV)
    for row in readCSV:
        #Get data from row
        b_id=int(row[0])
        b_meter=int(row[1])
        b_reading=float(row[3])
        #Add the data to the dict
        building_dict[b_id][b_meter].append(b_reading)

averages=[]

#Calculate meter averages
for building, data in building_dict.items(): 
    i=0
    daily_consumption=0
    while(i<4):
        num_readings=len(data[i])
        if(num_readings>0):
            daily_avg=sum(data[i])/num_readings*24
            print("Daily avg for " + str(building) + " for meter " + str(i) + " is " + str(daily_avg))
            daily_consumption=daily_consumption+daily_avg
        i=i+1
    averages.append(daily_consumption)


building_data=[]

def getCode(usage):
    if(usage=="Education"):
        return 0
    elif(usage=="Office"):
        return 1
    elif(usage=="Lodging/residential"):
        return 2
    elif(usage=="Parking"):
        return 3
    elif(usage=="Retail"):
        return 4
    elif(usage=="Warehouse"):
        return 5
    elif(usage=="Public services"):
        return 6
    elif(usage=="Entertainment/public assembly"):
        return 7
    elif(usage=="Food sales and service"):
        return 8
    elif(usage=="Utility"):
        return 9
    elif(usage=="Services"):
        return 10
    else:
        return 11

#Get building metadata
with open('building_metadata.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    #Skip the first line
    next(readCSV)
    for row in readCSV:
        #Get data from row
        b_id=int(row[1])
        b_usage=row[2]
        b_usecode=getCode(b_usage)
        b_sqft=float(row[3])
        if(not row[4]):
            b_yearsbuilt=avg_year_built
        else:
            b_yearbuilt=int(row[4])
        b_yearsold=2019-b_yearbuilt
        if(not row[5]):
            b_floors=median_floor_count
        else:
            b_floors=int(row[5])
        #Add the data to the dict
        building_data.append([b_usecode,b_sqft,b_yearsold,b_floors])








x_data = building_data
y_data = averages

#Transform y_data into a horizontal array
y_data=y_data.values.ravel()


# Do the cross validation
cv_scores = []
for _ in range(100):
    #Create a Gaussian Naive Bayes model
    clf = tree.DecisionTreeRegressor()
    #Calculate error based on which scoring method was used
    if(scoring_method=='neg_mean_squared_error'):
        #Get NMSE
        NMSE=np.mean(cross_validate(clf, x_data, y_data, cv=2,scoring=scoring_method)['test_score'])
        #Calculate RMSE and add it to our list of CV scores
        MSE=NMSE*-1
        RMSE=math.sqrt(MSE)
        cv_scores.append(RMSE)
    else:
        #Get accuracy
        percent_correct=np.mean(cross_validate(clf, x_data, y_data, cv=2,scoring=scoring_method)['test_score'])
        #Calculate error rate
        percent_incorrect=1-percent_correct
        cv_scores.append(percent_incorrect)
        
print(column_name, np.mean(cv_scores), np.std(cv_scores))




    
