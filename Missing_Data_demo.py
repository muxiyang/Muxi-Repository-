from pandas import Series, DataFrame
import pandas as pd
import numpy as np 
import os

main_dir="/Users/newlife-cassie/Desktop/data"
csv_file="sample_missing.csv"

# IMPORTING DATA: SETTING MISSING VALUES (SENTINELS)--------------------

df= pd.read_csv(os.path.join(main_dir,csv_file))
df.head() #top 5 values 
df.head(10) # head(n) gives top n rows 
df[:10] # same as slicing 

df.tail (10) # tail(n) gives bottom n rows 

df['consump'].head(10).apply(type)  # apply function 'type'to top 10 rows of consump

## we Don't want string data. periods'.' are common place holders for missing 
## data in some programing languages (stata). so we need to create 
## missing value sentinels to adjust 

missing=['.','NA','NULL',' ']
df= pd.read_csv(os.path.join(main_dir,csv_file),na_values=missing)
df.head(10)

##set missing values to floats, which means numerical numbers

# MISSING DATA (USING SMALLER DATAFRAME)------------------------

# Repeat lists by multiplying 
[1,2,3]
[1,2,3]*3

# types of missing data 
None 
np.nan 

type (None)   #--> NoneType 
type (np.nan) #--> float--> numeric --> more efficient 

## create a sample data set 
zip1=zip([2,4,8],[np.nan,5,7],[np.nan,np.nan,22])
df1=DataFrame(zip1, columns=['a','b','c'])

## search for missing data using 
df1.isnull() #pandas method to find missing data 
np.isnan(df1) # numpy way
## DataFrame. tab --> display all object can do with DataFrame

## subset of columns 
cols=['a','c']
df1[cols]
df1[cols].isnull()

## subset of series 
df1['b'].isnull()

## find non-missing values 
df1.isnull()
df1.notnull()
df1.isnull()==df1.notnull()  #check if filled in correctly --> want all false

# FILLING IN OR DROPPING VALUES----------------------------------------

## pandas method 'fillna'
df1.fillna(999) #fill missing value with 999
df2=df1. fillna(999)

## pandas method 'dropna'
df1.dropna() #drop ROWS with ANY missing values 
df1. dropna (axis=0, how='any') #drop ROWS with ANY missing values 

df1. dropna(axis=1, how='any') #drop COLUMN with ANY missing values 

df1. dropna(axis=0, how='all')#drop ROWS with ALL missing values 

df. dropna(how="all") #drop ROWS with ALL missing values 

#SEEING ROWS WITH MISSING DATA
df3=df.dropna(how='all')
df3['consump'].isnull()
rows=df3['consump'].isnull()
df3[rows]