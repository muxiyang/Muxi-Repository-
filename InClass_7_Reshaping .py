from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from dateutil import parser   # use this to ensure dates are parsed correctly

main_dir="/Users/newlife-cassie/Desktop/PUBPOL590"
root = main_dir + "/Data_All/"

#Import data-----------------------------------------------
df = pd.read_csv(root + "sample_30min.csv", header=0, parse_dates=[1],date_parser=parser.parse)
# header=0 means header on row zero 
# parse_dates=[1]: dates is the second column of data 
# date_parser=parser.parse: use the parser function to parse dates 

df_assign = pd.read_csv(root + 'sample_assignments.csv', usecols=[0,1])
# usecols=[0,1]: only want the first two columns 

# merge----------------------------------------------------
df = pd.merge(df, df_assign)

# add/drop variables---------------------------------------
df['year']= df['date'].apply(lambda x:x.year)
#apply the lambda: self-defined function

# the date data is in a datetime format, thus can extract year from it
type(df['date'][0])

df['month'] = df['date'].apply(lambda x: x.month)
df['day'] = df['date'].apply(lambda x: x.day)
df['ymd'] = df['date'].apply(lambda x: x.date())

# daily aggregation-----------------------------------------
grp = df.groupby(['ymd', 'panid', 'assignment'])
df1=grp['kwh'].sum().reset_index()
#sum kwh for hourly into daily data 

# Pivot data---------------------------------------------------
# go from 'long' to 'wide'

#1. create column names 
#create strings names and denote consumption and dates 
#use termery experession: [true-expr(x)] if condition else false-exp(x) for x in list)

#df1['day_str']=['0'+str(v) if v<10 else str(v) in df1['date']]
#add 0 in front of single digits --> preserve the correct order 
#df1['kwh_ymd']='kwh_'+df1.year.apply(str)+' '+df1.month.apply(str)+'_'+ df1.day_str.apply(str)

#short-version 
df1['kwh_ymd']='kwh_'+df1['ymd'].apply(str)

#2. pivot! aka long to wide 
df1_piv=df1.pivot('panid','kwh_ymd','kwh')
#pivot(i(row), j(column),what goes into the box)

#clean up 
df1_piv.reset_index(inplace=True)
df1_piv.columns.name=None 
df1_piv 

# merge time invariant data---------------------------------------
df2=pd.merge(df_assign,df1_piv) #this attaching order looks better 

# export data for regression 
df2.to_csv(root+"07_kwh_wide.csv", sep=",", index= False)