#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


datatr = pd.read_csv("/Users/Niklas/Desktop/Master ESCP/ML Python/trainFINAL.csv")
datats = pd.read_csv("/Users/Niklas/Desktop/Master ESCP/ML Python/test1FINAL.csv")


# In[3]:


datatr


# In[4]:


## Replacing class -1 with 0, because easier to work with
datatr["label"].replace({-1: 0}, inplace=True)
datats["label"].replace({-1: 0}, inplace=True)


# In[5]:


#Checking Data
for i in datatr.columns:
    print ("---- %s ---" % i)
    print (datatr[i].value_counts())


# In[6]:


## Checking, if there are missing values
datatr.isnull().sum()


# In[7]:


## Split train - data into X and Y variable
X = datatr.drop(["label","visitTime","hour","purchaseTime"], axis=1)
Y = datatr["label"]


# In[8]:


## Modeling with Logistic Regression
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.33, stratify=Y)


# In[9]:


## Oversampling the data set, as one class is underrepresented
import collections
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
print('Original dataset shape %s' % Counter(Y_train))
sm = SMOTE(random_state=42, sampling_strategy=0.3)
X_res, Y_res = sm.fit_resample(X_train,Y_train)
rs = RandomUnderSampler(random_state =42, sampling_strategy=0.7)
X_res, Y_res = rs.fit_resample(X_res, Y_res)
print('Resample dataset shape %s' % Counter(Y_res))


# In[10]:


## Run Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_res, Y_res)
predictions = logmodel.predict(X_test)


# In[11]:


## Classification Report
from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))


# In[12]:


## Create confusion matrix for better understanding
from sklearn.metrics import confusion_matrix
confusion_matrix(predictions,Y_test)


# In[13]:


## Random Forest Model - Check with model before
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500, max_depth = 3, random_state = 42)
rf.fit(X_res, Y_res)
predictionsrf = rf.predict(X_test)


# In[14]:


print(classification_report(Y_test, predictionsrf))


# In[15]:


confusion_matrix(predictionsrf,Y_test)


# In[16]:


## Display ROC curves for both models
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
ax = plt.gca()
lr_curve = plot_roc_curve(logmodel, X_test, Y_test,ax=ax, alpha=0.8)
rf_curve = plot_roc_curve(rf, X_test, Y_test,ax=ax, alpha=0.8)
plt.show()


# In[17]:


## We choose the RandomForest model to predict the test set, as it has a higher AUC


# In[18]:


## Fit the model to test dataset
datats


# In[19]:


## Split test - data into X and Y variable
Xts = datats.drop(["label","visitTime","hour","purchaseTime"], axis=1)
Yts = datats["label"]


# In[20]:


## Predict probabilities based on the test data set
predictionsts = rf.predict_proba(Xts)
predictionsts


# In[21]:


## Create data frame out of predictions and rename columns
prediction_frame = pd.DataFrame(predictionsts)
nf = prediction_frame.rename(columns = {0:"No_Purchase",1:"Purchase"})
nf


# In[22]:


## Merge two datasets and only keep relevant columns
result = pd.concat([Xts, nf], axis=1, join='inner')
Final = result[["id","Purchase"]]
Final


# In[23]:


Final.to_csv("/Users/Niklas/Desktop/Master ESCP/ML Python/Final_Assignment_2_3_Purchase_Prob.csv")


# In[ ]:




