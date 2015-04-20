from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from dateutil import parser   # use this to ensure dates are parsed correctly
import statsmodels.api as sm 
import os

main_dir="/Users/newlife-cassie/Desktop/Task4/"

#######################################################
#                    Section 0                        #
#######################################################

#Import logit function-------------------------------- 
os.chdir(main_dir)
from logit_functions import * 

#import data------------------------------------------
df=pd.read_csv(main_dir+ "14_B3_EE_w_dummies.csv")
df=df.dropna()

tariffs=[v for v in pd.unique(df['tariff']) if v != 'E']
stimuli=[v for v in pd.unique(df['stimulus']) if v != 'E']
tariffs.sort()
stimuli.sort()

# Run logit--------------------------------------------
drop = [ v for v in df.columns if v.startswith ("kwh_2010")]
df_pretrial = df.drop(drop,axis=1)

for i in tariffs:
    for j in stimuli: 
        logit_results, df_logit=do_logit(df_pretrial,i,j,add_D=None, mc=False)

# Means Comparison with T-Test by hand--------------------
grp=df_logit.groupby('tariff')
df_mean=df_pretrial.groupby('tariff').mean().transpose()
df_mean.B-df_mean.E #just difference, don't know if significant or not

#t-test by hand 
df_s=grp.std().transpose()
df_n=grp.count().transpose().mean()
top=df_mean['B']-df_mean['E']
bottom=np.sqrt(df_s['B']**2/df_n['B']+df_s['E']**2/df_n['E'])
tstats=top/bottom
sig=tstats[np.abs(tstats)>2]
sig.name='t-stats'


#######################################################
#                    Section 1                        #
#######################################################

#Test for Imbalance running logit-------------------------------
df=pd.read_csv(main_dir+ "/task_4_kwh_w_dummies_wide.csv")
df=df.dropna()

tariffs=[v for v in pd.unique(df['tariff']) if v != 'E']
stimuli=[v for v in pd.unique(df['stimulus']) if v != 'E']
tariffs.sort()
stimuli.sort()

# Run logit
drop = [ v for v in df.columns if v.startswith ("kwh_2010")]
df_pretrial = df.drop(drop,axis=1)

for i in tariffs:
    for j in stimuli: 
        logit_results, df_logit=do_logit(df_pretrial,i,j,add_D=None, mc=False)
        
# Test for Imbalance with a "Quick Means Comparison"------------
grp=df_logit.groupby('tariff')
df_mean=df_pretrial.groupby('tariff').mean().transpose()
df_mean.C-df_mean.E #just difference, don't know if significant or not

#t-test by hand 
df_s=grp.std().transpose()
df_n=grp.count().transpose().mean()
top=df_mean['C']-df_mean['E']
bottom=np.sqrt(df_s['C']**2/df_n['C']+df_s['E']**2/df_n['E'])
tstats=top/bottom
sig=tstats[np.abs(tstats)>2]
sig.name='t-stats'

#######################################################
#                    Section 2                        #
#######################################################
logit_results,df_logit=do_logit(df_pretrial,'C','4',add_D=None,mc=False)

df_logit['p_val'] = logit_results.predict()
df_logit['trt']=0+ (df_logit['tariff']=='C')

df_logit['w']=np.sqrt(df_logit['trt']/df_logit['p_val']+(1-df_logit['trt'])/df_logit['p_val'])

df_w=df_logit[['ID','trt','w']]

#######################################################
#                    Section 3                        #
#######################################################

# 1.import fe_function and data--------------------------
os.chdir(main_dir)
from fe_functions import * 

df=pd.read_csv(main_dir+ "task_4_kwh_long.csv")

# 2. merge df_w and df2----------------------------------
df2=pd.merge(df,df_w)

# 3. create necessary variables---------------------------
df2['trtint']=df2['trt']*df2['trial']
df2['log_kwh'] = (df2['kwh'] + 1).apply(np.log)
# create month string `mo_str` that adds "0" to single digit integers
df2['mo_str'] = np.array(["0" + str(v) if v < 10 else str(v) for v in df2['month']])
# concatenate to make ym string values
df2['ym'] = df2['year'].apply(str) + "_" + df2['mo_str']

# 4. set up regression variables----------------------------
y=df2['log_kwh']
P=df2['trial'] 
TP=df2['trtint']
w=df2['w']
mu = pd.get_dummies(df2['ym'], prefix = 'ym').iloc[:, 1:-1]

X = pd.concat([TP, P, mu], axis=1)

#5. De-mean y and x-------------------------------------------
ids = df2['ID']
y = demean(y, ids)
X=demean(X,ids)

#6. Run Fixed Effects-------------------------------------------

## WITHOUT WEIGHTS
fe_model = sm.OLS(y, X) # linearly prob model
fe_results = fe_model.fit() # get the fitted values
print(fe_results.summary()) # print pretty results (no results given lack of obs)

# WITH WEIGHTS
## apply weights to data
y = y*w # weight each y
nms = X.columns.values # save column names
X = np.array([x*w for k, x in X.iteritems()]) # weight each X value
X = X.T # transpose (necessary as arrays create "row" vectors, not column)
X = DataFrame(X, columns = nms) # update to dataframe; use original names

fe_w_model = sm.OLS(y, X) # linearly prob model
fe_w_results = fe_w_model.fit() # get the fitted values
print(fe_w_results.summary()) # print pretty results (no results given lack of obs)

