#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:40:54 2020

@author: Vineeth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

raw_data = pd.read_csv('/Users/Vineeth/Documents/Studies/The Data Science Course 2019 - All Resources/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S35_L227/1.04. Real-life example.csv')
raw_data.head()

x = raw_data.describe(include='all')

data = raw_data.drop(['Model'], axis=1)

data.isnull().sum()
data_no_null = data.dropna(axis=0)
data_no_null.isnull().sum()

x=data_no_null.describe(include='all')

sns.distplot(data_no_null['Price'])

y = data_no_null['Price'].quantile(0.99)

data_1 = data_no_null[data_no_null['Price']<y]

sns.distplot(data_1['Price'])
x=data_1.describe(include='all')

sns.distplot(data_1['Mileage'])
y=data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<y]

x=data_2.describe(include='all')

sns.distplot(data_2['Mileage'])

sns.distplot(data_2['EngineV'])

data_3 = data_2[data_2['EngineV']<6.5]
sns.distplot(data_3['EngineV'])

sns.distplot(data_3['Year'])

q = data_3['Year'].quantile(0.01)

data_4 = data_3[data_3['Year']>q]
sns.distplot(data_4['Year'])

data_cleaned = data_4.reset_index(drop=True)

x=data_cleaned.describe(include='all')

m = data_cleaned['Year']
n = data_cleaned['Price']

#plt.scatter(m,n)
#plt.xlabel('Year', fontsize = 20)
#plt.ylabel('Price', fontsize = 20)
#plt.show()

f, (ax1,ax2,ax3) = plt.subplots(1,3,sharey = True, figsize = (15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')
plt.show()

sns.distplot(data_cleaned['Price'])

log_price = np.log(data_cleaned['Price'])

data_cleaned['log_price'] = log_price

data_cleaned

f, (ax1,ax2,ax3) = plt.subplots(1,3,sharey = True, figsize = (15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log_Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
ax2.set_title('Log_Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
ax3.set_title('Log_Price and Mileage')
plt.show()

data_cleaned = data_cleaned.drop(['Price'],axis=1)

data_cleaned.columns.values

#Below code is to check multicollinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = data_cleaned[['Mileage','Year','EngineV']]

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns

vif

data_no_multicollinearity = data_cleaned.drop(['Year'], axis = 1)

data_no_multicollinearity

#variables1 = data_no_multicollinearity[['Mileage','EngineV']]

#vif1 = pd.DataFrame()

#vif1['VIF'] = [variance_inflation_factor(variables1.values,i) for i in range(variables1.shape[1])]
#vif1['Features'] = variables1.columns

#vif1

#Introducing dummy variables

data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
data_with_dummies.columns.values

cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']

data_preprocessed = data_with_dummies[cols]

variables = data_preprocessed.drop(['log_price'],axis=1)
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
vif

#Linear Regression Model

targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis=1)

#Scale the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)

scaled_input = scaler.transform(inputs)

#Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(scaled_input,targets,test_size = 0.2,random_state=42)

reg = LinearRegression()
reg.fit(x_train,y_train)

y_hat = reg.predict(x_train)

#Checking the model's accuracy
plt.scatter(y_train,y_hat)
plt.xlabel('Targets',fontsize=20)
plt.ylabel('Predictions',fontsize=20)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

#Another method to check accuracy
#By residual plt
sns.distplot(y_train - y_hat)
plt.title("Residuals PDF", size = 18)


reg.score(x_train,y_train)

def adj_r2(x,y):
    r2 = reg.score(x_train,y_train)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2

adj_r2(x_train,y_train)

reg.intercept_
reg.coef_

reg_summary = pd.DataFrame(inputs.columns.values,columns = ['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary

data_cleaned['Brand'].unique()
data_cleaned['Body'].unique()
data_cleaned['Engine Type'].unique()
inputs.columns.values

#Testing our model


























































