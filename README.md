# House Price Prediction with Machine Learning Model

The aim of this project is to identify the suitable model to make the prediction for the house price with given significant predictor variables and used a supervised learning technique. Two files, train and test are provided and the price of the test data is to be estimated. Here I have used XGBoost for prediction.


## Steps in House Price Prediction
1. Import the required libraries
2. Load the data
3. Data Exploration
4. Data Cleaning
5. Data transformation
6. Base Model Performance (XGBoost)
7. Hyperparameter Tuning
8. Final Model
9. Visualize Results

## 1. Import the required libraries
```
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
## 2. Load the data
```
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
```
 ## 3. Data Exploration
 ```
 df.head()
df_test.shape
df.shape
# Checking the null values
df.isnull().sum()
df_test.isnull().sum()
# heatmap for visualizing the null vaues
sns.heatmap(df.isnull(),yticklabels = False,cbar = False)
df.info()
```
## 4. Data Cleaning
```
# Handling missing data for MSZoning
df['MSZoning'].value_counts()
df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0]) # replacing with mode
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean()) # replacing with mean for train
df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].mean()) # replacing with mean for train
```
