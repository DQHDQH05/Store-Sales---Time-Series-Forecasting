#!/usr/bin/env python
# coding: utf-8

# # Shuqi Lin -- 6.26

# In[1]:


#%% 1. Packages
import numpy as np
import pandas as pd
import os
import gc
import warnings

# PACF - ACF
# ------------------------------------------------------
import statsmodels.api as sm

# DATA VISUALIZATION
# ------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[4]:


# CONFIGURATIONS
# ------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')


# In[8]:


# 2. Import Data
## Make sure you are under the folder with all datasets
train=pd.read_csv('train_clean_Ziyi.csv')
train.drop('Unnamed: 0',axis=1,inplace=True)
test = pd.read_csv("test.csv",parse_dates=['date'])
stores = pd.read_csv("stores.csv")
#sub = pd.read_csv("../input/store-sales-time-series-forecasting/sample_submission.csv")   
transactions = pd.read_csv("transactions.csv",parse_dates=['date']).sort_values(["store_nbr", "date"])


# In[9]:


os.chdir('.\\clustered training data')
n_clusters = len(stores['cluster'].unique())
for i in range(n_clusters):
    cluster_df=d.loc[d['store_nbr'].isin(list(stores.iloc[list(stores.groupby('cluster').groups[i+1])]['store_nbr']))]
    cluster_df.to_csv('cluster_'+str(i+1)+'.csv',index=False)

os.chdir('..\\clustered testing data')
for i in range(n_clusters):
    cluster_df=test.loc[test['store_nbr'].isin(list(stores.iloc[list(stores.groupby('cluster').groups[i+1])]['store_nbr']))]
    cluster_df.to_csv('cluster_'+str(i+1)+'.csv',index=False)


# In[ ]:




