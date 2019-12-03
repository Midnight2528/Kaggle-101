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
        
