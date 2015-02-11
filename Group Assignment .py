from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np 
import os

main_dir="/Users/newlife-cassie/Desktop/Group_Assignment_Data"
txt1="File1.txt"
txt2="File2.txt"
txt3="File3.txt"
txt4="File4.txt"
txt5="File5.txt"
txt6="File6.txt"

#DEFINE DATAFRAMES WITH MISSING DATA----------------------------------------
missing=['.','NA','NULL',' ','0','-','999999999']
df1= pd.read_table(os.path.join(main_dir,txt1),sep=" ",header=None,names=["mid","time","reading"],na_values=missing)
df2= pd.read_table(os.path.join(main_dir,txt2),sep=" ",header=None,names=["mid","time","reading"],na_values=missing)
df3= pd.read_table(os.path.join(main_dir,txt3),sep=" ",header=None,names=["mid","time","reading"],na_values=missing) 
df4= pd.read_table(os.path.join(main_dir,txt4),sep=" ",header=None,names=["mid","time","reading"],na_values=missing)
df5= pd.read_table(os.path.join(main_dir,txt5),sep=" ",header=None,names=["mid","time","reading"],na_values=missing)
df6= pd.read_table(os.path.join(main_dir,txt6),sep=" ",header=None,names=["mid","time","reading"],na_values=missing)
##no header, delimiter is space,defined column names


#MERGE 6 DATAFRAMES-----------------------------------------
df=pd.concat([df1,df2,df3,df4,df5,df6], axis=0,ignore_index=True)
## row bind with keys add together 
##shape(df):dimention of the new dataframe
##len(df): number of items in the new dataframe 

# MISSING DATA
pd.isnull(df1).any(1).nonzero()[0]
## confirm NaN

# DUPLICATED DATA
    
b.drop_duplicates(['mid','time'],take_last=1) #? kept one of the duplicates? 

# Dropping row values 
df1=df.copy()
df1.drop(range(0,9),inplace=True)

#Change row values 

df.['kwh'][if a number is more than 49]=NaN
