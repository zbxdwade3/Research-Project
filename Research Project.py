#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##### Part 1: Load data
#####   This part takes the raw data from a json file, and then analyse the dimensions of the data
##### Part 2: Data selection
#####   This part extract data to be analysed in the next section
##### Part 3: Formaing a pipleline 
#####   This part put data preprocessing, parameter selection and model fitting into a pipeline

###     Part 1. Load data

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import json
with open('/Users/yourongzhang/Downloads/MIMIC3/mimic3_full_data_dx_index.json') as f:
    data=json.load(f)

##### Run first loop to get diagnose max index and decide how many rows of data is usable
lines=0;# Count how many rows of data can be used
maxDiagindex=0; # Browse through the diagnose and find the max index number, 
                # so to determine how to shape the diagnose matrix with zeros
for eachpatient in data:
    for eachadmin in eachpatient['visits']:
        try:
            if (max(eachadmin['DXs'])>maxDiagindex): maxDiagindex=max(eachadmin['DXs'])
            lines +=1;
        except:
            pass

###     Part 2. Data selection

####### 2.1 Retreiving data: 
#1. diagarray - (eg. =array(19939, 2438), each row represent a patient admission, each column is 0 or 1 
#                , for example '1' in the 3rd colum represent the patient has found sympton 3) 
#2. dayarray - (eg. =array(19939,1),each row represent a patient admission, column represent how many days of stay)
#3. moralarray - (ed.=array(19939,1),each row represent a patient admissionm, column represent morality)

# Store the diagnose into 0 arrays, with each column represent a diagnose, 1 is found, 0 is not found
diagarray = np.zeros((lines,maxDiagindex+1))
dayarray = np.zeros(lines);
moralarray = np.zeros(lines);
readmissionraw=[];
index = 0;
for eachpatient in data:
    for eachadmin in eachpatient['visits']:
        if(eachadmin['DXs']==[]):
            continue;
        else:
            moralarray[index]=eachadmin['Death']
            dayarray[index]=eachadmin['day_cnt'];
            for eachDXs in eachadmin['DXs']:
                try:
                    diagarray[index,eachDXs]=1;
                except:
                    pass
            readmissionraw.append((int(eachpatient['pid']),eachadmin['admsn_dt'],eachadmin['day_cnt'],diagarray[index]))
        index+=1;
# 1. diagarray is obtained 2. dayarray is obtained 3. moralarray is obtained

# Based on user requirement, we can further process the dayarray for the purpose of determining:
# 1. Resources needs to be allocated for this patient if stay is longer than 7
# 2. The patient can leave on the same day
userprescribedday = 7;###### This value can be any USER INPUT value
less7=[];
for eachday in dayarray:
    if (eachday>userprescribedday):
        less7.append(1)
    else:
        less7.append(0)
# less7 is an array, where each row is a patient admission and (1/0)represents whether the patient stayed longer 
# than the prescirbed number of days

######## 2.2 Obtain a new array for predicting whether readmission will happen
# This arrary must contain lasttime diagnose and time difference (between this admission and last admission)
# Retrieve data 
# 1.myarray  - each row is a patient admission, colums are: patient id, date of admission, days of stay, diagnose array
# 2.diagforlastadmin - diagnose array for the same patient but last admission
# 3.overlimitdayornot - single column of each patient readmission, saying whether this admission to last admission is over 30 days

# Retrive data and sort the data based on patient id and admission date
import pandas as pd
mydataframe = pd.DataFrame(readmissionraw)
mydataframe.columns=['pid','date','day_cnt','diagnose']
mydataframe = mydataframe.sort_values(by=['pid','date'],ascending=True)
myarray = np.array(mydataframe)

# By comparing admission date of each patient, get the time how long the patient has not come to hospital
# and get the diagnose of last admission
from datetime import datetime
index = 0;
readmission=[];
buffer=None;
firstflag=0;
for lines in myarray:
    myarray[index][1] = datetime.strptime(myarray[index][1],'%Y%m%d')
    index +=1;
    if(firstflag==0):#get the first line as a buffer to initialise comparison with the second line
        firstflag=1;
        buffer = lines;
    else:
        #same patient
        if(buffer[0]==lines[0]):
            #if same date, put diagnose together and use the second one for storage
            if(lines[1]-buffer[1]==0):
                # need to store the combined lines[2]
                index2=0; # to bitwise add the 
                for eachdiagnosebit in buffer[3]:
                    lines[3][index2] = np.bitwise_or(int(lines[3][index2]),int(eachdiagnosebit))
                    index2+=1;
                #when there are patient coming in the same day more than once
                #it should be stored in buffer and wait for the next same pid but different day
                buffer=lines;
            else:#if different date
                readmission.append((lines[0],(lines[1]-buffer[1]).days,lines[3]))
                buffer=lines;
        #different patient
        else:
            buffer = lines;

# diagnose of last admission
diagforlastadmin = np.zeros((len(readmission),maxDiagindex))
overlimitdayornot =[];
index=0;
readmissionlimits = 30; #re admission in 30 days//////////this value can be changed 
for each in readmission:
    diagforlastadmin[index]=each[0];
    if (each[1]>readmissionlimits):
        overlimitdayornot.append(1)
    else:
        overlimitdayornot.append(0)           
#X  - diagforlastadmin
#y  - overlimitdayornot

# Question 1:   X - diagnose               y - morality
# Question 2,3: X - diagnose               y - length of stay
# Question 4:   X - diagnose(reshaped)     y - readmission within 30 days

# Corresponding X,y in each question:
# Q1: X = diagarray          y = moralarray
# Q2: X = diagarray          y = less7
# Q3: X = diagarray          y = dayarray
# Q4: X = diagforlastadmin   y = overlimitdayornot

# Split data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diagarray,moralarray ,test_size=0.33, random_state=19)


###     Part 3. Forming piepline

# Normalise and dimension reduction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
mypipe = Pipeline(steps=[('norm',StandardScaler()),('pca',PCA()),('clf',KNeighborsClassifier())])
mypipe.fit(X_train,y_train)
mypipe.score(X_test,y_test)

parameters = { 
    'clf__n_neighbors': [3,5,7,9]}
    #'clf__weights' :['uniform','distance']}

from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(mypipe, parameters, n_jobs= -1)
CV.fit(X_train, y_train)  


# set the parameters to the classifier
#knn = KNeighborsClassifier(n_neighbors=gsresult.best_params_['n_neighbors'],weights=gsresult.best_params_['weights'])
#knn.fit(X_train, y_train)

