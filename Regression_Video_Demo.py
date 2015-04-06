from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import statsmodels.api as sm
from dateutil import parser   # use this to ensure dates are parsed correctly

main_dir="/Users/newlife-cassie/Desktop/PUBPOL590"
root = main_dir + "/Data_All/"

## IMPORT DATA-----------------
df=pd.read_csv(root+"07_kwh_wide.csv",header=0)

#SIMPLE LINEAR PROBABILITY MODEL(LPM)
##lets see if consumption before a certain date determined your assignment 

df['T']=0+(df['assignment']=='T')
#make a dummy variable for treatment assignment 
#keep boolean statements in ()

## SET UP DATA 
# get X matrix (left hand variables for our regression)
kwh_cols=[v for v in df.columns.values if v.startswith('kwh')] 

#pretend that the treatment occured in 2015-01-04. We want the dates before 

#'kwh-2015-01-01'is a string value--> v[0:3] gives 'kwh', v[-1:] gives 1

kwh_cols=[v for v in kwh_cols if int(v[-2:])<4]

#set up y and x 
y=df['T']
X=df[kwh_cols]

X=sm.add_constant(X)

# RUN OLS 
ols_model=sm.OLS(y,X) 
ols_results= ols_model.fit() #fit the model 
print(ols_results.summary())


