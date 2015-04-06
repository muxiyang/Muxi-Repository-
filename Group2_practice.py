from __future__ import division
from pandas import Series, DataFrame
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt

main_dir="/Users/newlife-cassie/Desktop"

# ADVANCED PATHING---------------------------------------------------------------
root = main_dir + "/Group_Assignment_Data/"
paths = [root + v for v in os.listdir(root) if v.startswith("File1")]

#IMPORT DATA & Stacking-----------------------------------------------------------
missing=['.','NA','NULL',' ','-','999999999']
list_of_dfs = [pd.read_table(v, names = ['ID', 'time', 'kwh'], sep = " ", na_values= missing) for v in paths]
df_stack = pd.concat (list_of_dfs, ignore_index = True)
del list_of_dfs

df_assign = pd.read_csv(root + "SME and Residential allocations.csv", na_values= missing, usecols = [0,1,2,3])
df_assign.columns = ['ID', 'code', 'tariff', 'stimulus']

# MERGING------------------------------------------------------------------------
df = pd.merge(df_stack, df_assign)

# NEW Date Variables 
df['hour_cer'] = df['time'] % 100
df['day_cer'] = (df['time'] - df['hour_cer']) / 100
df.sort(['ID', 'time'], inplace = True)

# TRIMMING 
df = df[df['code'] == 1]
df = df[(df['stimulus'] == 'E') | ((df['stimulus'] == '1') & (df['tariff'] == 'A'))]

# Examing the data
df1 = df[(df['tariff'] == 'A') & (df['stimulus'] != '1')] # checking if there are people only assigned tariff A but not stimulus 1 "bi-monthy bill"
df2 = df[(df['tariff'] != 'A') & (df['stimulus'] == '1')] # checking if there are people only stimulus 1 but not tariff A
df3 = df[(df['tariff'] == 'E') & (df['stimulus'] != 'E')] 
df4 = df[(df['tariff'] != 'E') & (df['stimulus'] == 'E')] # checking if people are assigned in control group are in the control in both treatment
df5 = df[df['tariff'] == 'A']

# TIME SERIES CORRECTION ---------------------------
df_time = pd.read_csv(root + "timeseries_correction.csv", na_values = missing, header = 0, parse_dates = [1], usecols = [1,2,9,10])
# We can just use the csv data as well, code: df_time = pd.read_csv(root + "timeseries_correction.csv", na_values = missing, header = 0, parse_dates = [1], usecols = range(1,11))
df = pd.merge(df, df_time)

# ADD/DROP VARIABLES ---------------------------
df['year'] = df['date'].apply(lambda x: x.year)
df['month'] = df['date'].apply(lambda x: x.month)
df['day'] = df['date'].apply(lambda x: x.day)

# MONTHLY AGGREGATION --------------------
grp = df.groupby(['year', 'month', 'ID', 'tariff'])
agg = grp['kwh'].sum()

# reset the index (multilevel at the moment)
agg = agg.reset_index() # drop the multi-index
grp1 = agg.groupby(['year', 'month', 'tariff'])


# Get separate sets of treatment and control values by date
trt = {(k[0],k[1]): df.kwh[v].values for k, v in grp1.groups.iteritems() if k[2] == 'A'}
ctrl = {(k[0],k[1]): df.kwh[v].values for k, v in grp1.groups.iteritems() if k[2] == 'E'}

# create dataframes of this information
keys = trt.keys()
tstats = DataFrame([(k, np.abs(ttest_ind(trt[k],ctrl[k], equal_var=False)[0])) for k in keys],
    columns =['ymd', 'tstats'])
pvals = DataFrame([(k, (ttest_ind(trt[k],ctrl[k], equal_var=False)[1])) for k in keys],
    columns =['ymd', 'pvals'])
t_p = pd.merge(tstats, pvals)

## sort and reset _index
t_p.sort(['ymd'], inplace=True) # inplace = True to change the values
t_p.reset_index(inplace=True, drop=True)
# t_p = t_p.dropna(axis = 0, how = 'any')  # drop any missing values in tstats and pvals

# PLOTTING ----------------------
fig1 = plt.figure() #initialize plot
ax1 = fig1.add_subplot(2,1,1) # (row, columns, reference) two rows, one column, first plot
ax1.plot(t_p['tstats'])
ax1.axhline(2, color='r', linestyle ='--')
ax1.axvline(6, color='g', linestyle ='--')
ax1.set_title('t-stats over-time (monthly)')

ax2 = fig1.add_subplot(2,1,2) # (row, columns, reference) two rows, one column, second plot
ax2.plot(t_p['pvals'])
ax2.axhline( 0.05, color='r', linestyle ='--')
ax2.axvline(6, color='g', linestyle ='--')
ax2.set_title('p-values over-time (monthly)')

# DAILY AGGREGATION --------------------
grp = df.groupby(['year', 'month','date', 'ID', 'tariff'])
agg = grp['kwh'].sum()

# reset the index (multilevel at the moment)
agg = agg.reset_index() # drop the multi-index
grp1 = agg.groupby(['year', 'month','date', 'tariff'])


# Get separate sets of treatment and control values by date
trt = {(k[0], k[1],k[2]): df.kwh[v].values for k, v in grp1.groups.iteritems() if k[3] == 'A'}
ctrl = {(k[0], k[1],k[2]): df.kwh[v].values for k, v in grp1.groups.iteritems() if k[3] == 'E'}

# create dataframes of this information
keys = trt.keys()
tstats = DataFrame([(k, np.abs(ttest_ind(trt[k],ctrl[k], equal_var=False)[0])) for k in keys],
    columns =['date', 'tstats'])
pvals = DataFrame([(k, (ttest_ind(trt[k],ctrl[k], equal_var=False)[1])) for k in keys],
    columns =['date', 'pvals'])
t_p = pd.merge(tstats, pvals)

## sort and reset _index
t_p.sort(['date'], inplace=True) # inplace = True to change the values
t_p.reset_index(inplace=True, drop=True)
##t_p = t_p.dropna(axis = 0, how = 'any')  # drop any missing values in tstats and pvals

# PLOTTING ----------------------
fig1 = plt.figure() #initialize plot
ax1 = fig1.add_subplot(2,1,1) # (row, columns, reference) two rows, one column, first plot
ax1.plot(t_p['tstats'])
ax1.axhline(2, color='r', linestyle ='--')
ax1.axvline(180, color='g', linestyle ='--')
ax1.set_title('t-stats over-time (daily)')

ax2 = fig1.add_subplot(2,1,2) # (row, columns, reference) two rows, one column, second plot
ax2.plot(t_p['pvals'])
ax2.axhline( 0.05, color='r', linestyle ='--')
ax2.axvline( 180, color='g', linestyle ='--')
ax2.set_title('p-values over-time (daily)')