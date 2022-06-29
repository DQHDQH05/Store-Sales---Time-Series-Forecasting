#MAIN
import numpy as np
import pandas as pd
import math
import os
import gc
import warnings
import statsmodels.api as sm
import csv

# DATA VISUALIZATION
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# CONFIGURATIONS
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')

#import
train = pd.read_csv(r"c:\Users\Colin\Desktop\Kaggle\Sale_Prediction\train_Ziyi.csv")
oil = pd.read_csv(r"c:\Users\Colin\Desktop\Kaggle\Sale_Prediction\oil.csv")

#Fill the missing oil price 
previous_price = 93.14
for index, row in oil.iterrows():
    if math.isnan(row['dcoilwtico']):
        oil.loc[oil['date'] == row['date'], 'dcoilwtico'] = previous_price
    else:
        previous_price = row['dcoilwtico']
oil.to_csv('oil_filled.csv')


###7 day moving average
calendar = pd.DataFrame(index=pd.date_range('2013-01-01', '2017-08-31'))
oil['avg_oil'] = oil['dcoilwtico'].rolling(7).mean()
oil.to_csv('oil_seven.csv')

###30 day moving average 
#oil['avg_oil'] = oil['dcoilwtico'].rolling(30).mean()
#oil.to_csv('oil_thrity.csv')

###oil lag
oil_lag_time = [1,2,3,4,5,6,7,14,28,60,120,180,365]
oil_lag = oil.copy()
for lag in oil_lag_time:
    oil_lag[f"lag_{lag}"] = oil_lag['dcoilwtico'].transform(lambda x: x.shift(lag))
oil_lag.to_csv('oil_lag.csv')