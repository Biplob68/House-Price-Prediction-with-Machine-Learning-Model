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

Dropping the columns which have lot of missing values
```
df.drop(['PoolQC'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for train
df_test.drop(['PoolQC'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for test
df.drop(['Fence'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for train
df_test.drop(['Fence'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for test
df.drop(['MiscFeature'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for train
df_test.drop(['MiscFeature'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for test
df.drop(['FireplaceQu'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for train
df_test.drop(['FireplaceQu'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for test

```
```
## 5. Data Transformation
```
# function to convert categorical variables to one hot encoding
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final

```
```
# making a copy of dataframe for future use
main_df=df.copy()
# concanating the test and train files to implement one hot encoding
final_df=pd.concat([df,df_test],axis=0)
final_df=category_onehot_multcols(columns)
```
```
# removing duplicated columns
final_df =final_df.loc[:,~final_df.columns.duplicated()]
```
```
# separating the test and training data
df_Train=final_df.iloc[:1459,:]
df_Test=final_df.iloc[1459:,:]
```
```
# dropping the "SalePrice" column from test data
df_Test.drop(['SalePrice'],axis=1,inplace=True)
```
```
# preparing data for feeding into model
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
```
