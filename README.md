# House Price Prediction with Machine Learning Model

The aim of this project is to identify the suitable model to make the prediction for the house price with given significant predictor variables and used a supervised learning technique. Two files, train and test are provided and the price of the test data is to be estimated. Here I have used XGBoost for prediction.


## Steps in House Price Prediction
1. Import the required libraries
2. Load the data
3. Data Cleaning
4. Data transformation
5. Base Model Performance (XGBoost)
6. Hyperparameter Tuning
7. Final Model
8. Visualize Results

## 1. Import the required libraries
```
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
## 2.Load the data




```
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
```
 
