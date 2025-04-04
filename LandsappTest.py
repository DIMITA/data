#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')



# In[2]:


df= pd.read_csv('Basepourmodele.csv', encoding='latin-1')


# In[3]:


df.describe()


# In[4]:


df.head()



# In[5]:


df.tail()


# In[6]:


#df.plot(kind='box', subplots=False, layout=(10,10), sharex=False, sharey=False)
#plt.show()


# In[7]:


df = df.reset_index()


# In[8]:


pd.plotting.scatter_matrix(df, alpha=0.2)
plt.show()


# In[9]:


#sns.heatmap(df.corr(),annot=True)


# In[10]:


features = df[['N','P','K','Mg','pH','Temp','Pluviometrie','Humidite']]
target = df['Nom commun']


# In[11]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.20,random_state= 1)


# In[12]:


#Modèle Random Forest



RF = RandomForestClassifier(n_estimators=20, random_state=0)

RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)

print("Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[27]:


#Faire des prédictions
#'N/ha (en Kg)', 'P/ha (en Kg)','K/ha (en Kg)','Mg/ha (en Kg) ', 'Urée/ha (en Kg)', 'Fumier/ha (en t)', 'pH-sol', 'Temperature (en °C)', 'Pluviometrie (en mm)', 'Humidite (en %)'


données = np.array([[120, 40, 54, 40, 7, 33, 1700, 79]])
prediction = RF.predict(données)
print(prediction)


# In[14]:


from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)

print("DecisionTrees's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[15]:


from sklearn.svm import SVC

SVM = SVC(gamma='auto')

SVM.fit(Xtrain,Ytrain)

predicted_values = SVM.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)

print("SVM's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[16]:


from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(random_state=2)

LogReg.fit(Xtrain,Ytrain)

predicted_values = LogReg.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)

print("Logistic Regression's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[17]:


# Tester éventuellement Xgboost, LightGBM et faire des traitements de données pour faire marcher LogisticRegression et regarder les sorties à plusieurs options, prédiction conformale


# In[21]:


#Faire des prédictions
#Nom commun	N	P	K	Mg	pH	Temp	Pluviometrie	Humidite



données = np.array([[120, 40, 54, 40, 7, 33, 1700, 79]])
prediction = LogReg.predict(données)
print(prediction)


# In[18]:


from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)

print("Naive Bayes's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[23]:


#Faire des prédictions
#Nom commun	N	P	K	Mg	pH	Temp	Pluviometrie	Humidite



données = np.array([[120, 40, 54, 40, 7, 33, 1700, 79]])
prediction = NaiveBayes.predict(données)
print(prediction)


# In[19]:


#!pip install xgboost
import xgboost as xgb
XB = xgb.XGBClassifier()
XB.fit(Xtrain.astype('float'),Ytrain)

predicted_values = XB.predict(Xtest.astype('float'))

x = metrics.accuracy_score(Ytest, predicted_values)

print("XGBoost's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[22]:


#Faire des prédictions
#Nom commun	N	P	K	Mg	pH	Temp	Pluviometrie	Humidite



données = np.array([[120, 40, 54, 40, 7, 33, 1700, 79]])
prediction = XB.predict(données)
print(prediction)


# In[20]:


#!pip install lightgbm

import lightgbm as ltb

modelltb = ltb.LGBMClassifier()

modelltb.fit(Xtrain.astype('float'),Ytrain)

predicted_values = modelltb.predict(Xtest.astype('float'))

x = metrics.accuracy_score(Ytest, predicted_values)

print("Lightgbm's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[28]:


#Faire des prédictions
#Nom commun	N	P	K	Mg	pH	Temp	Pluviometrie	Humidite



données = np.array([[120, 40, 54, 40, 7, 33, 1700, 79]])
prediction = modelltb.predict(données)
print(prediction)


# In[ ]:




