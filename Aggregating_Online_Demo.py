from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

main_dir="/Users/newlife-cassie/Desktop"
root = main_dir + "/Group_Assignment_Data/"

df = pd.read_csv(root + "sample_30min.csv", header=0, parse_dates=[1])
df_assign = pd.read_csv(root + 'sample_assignments.csv', usecols=[0,1])

df = pd.merge(df, df_assign)

df['year'] = df['date'].apply(lambda x: x.year)
df['month'] = df['date'].apply(lambda x: x.month)
df['day'] = df['date'].apply(lambda x: x.day)

grp = df.groupby(['year', 'month', 'day', 'panid', 'group'])
agg = grp['kwh'].sum()

agg = agg.reset_index()
grp1 = agg.groupby(['year', 'month', 'day', 'group'])

#Split into T/C 

trt = {(k[0], k[1], k[2]): agg.kwh[v].values 
    for k, v in grp1.groups.iteritems() if k[3] == 'T'}
ctrl = {(k[0], k[1], k[2]): agg.kwh[v].values 
    for k, v in grp1.groups.iteritems() if k[3] == 'C'}
keys = ctrl.keys()

tstats = DataFrame([(k, np.abs(float(ttest_ind(trt[k], ctrl[k], equal_var=False)[0])))
    for k in keys], columns=['ymd', 'tstat'])
pvals = DataFrame([(k, (ttest_ind(trt[k], ctrl[k], equal_var=False)[1]))
    for k in keys], columns=['ymd', 'pval'])
t_p = pd.merge(tstats, pvals)


t_p.sort(['ymd'], inplace=True)
t_p.reset_index(inplace=True, drop=True)