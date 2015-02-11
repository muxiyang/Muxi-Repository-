from pandas import Series, DataFrame
import pandas as pd
import numpy as np 
import os

# From Group Assignment 

## TIME HOW LONG THE CODE TAKES
##start=time.time()
##print
#df.sort (['variable'],inplace=true)

# Class 5

main_dir="/Users/newlife-cassie/Desktop/"
root=main_dir+"Class5_Data/"

# PATHING 
paths=[os.path.join(root,v) for v in os.listdir(root) if v. startswith("file_")]

#IMPORT and STACK 
df=pd.concat([pd.read_csv(v, names=['panid','date','kwh']) for v in paths],ignore_index=True)

# IMPORT and MERGE
df_assign=pd.read_csv(root+"sample_assignments.csv",usecols=[0,1])
df=pd.merge(df,df_assign)

#GROUPBY aka "split,apply,combine"

## see more at the website 

#split by c/T, pooled without time 

#splitting by assignment 
groups1=df.groupby(['assignment'])  #just group by assignment, don't care about time, same as collapse in STATA
groups1.groups    # result is a index (dictionary) of treatment and control group 

#apply the mean 
groups1['kwh'].apply(np.mean)   #cannot see the things in the groups1, just calaculate the mean 
#the result is a cross-sectional (no time indicator) mean 

groups1['kwh'].mean() #internal function, faster --> use this!!!

%timeit -n 100 groups1['kwh'].apply(np.mean) #time how long it takes to calculate 
%timeit -n 100 groups1['kwh'].mean() #time how long it takes to calculate 
#the internal function is faster 

#Split by control and treatment but pooling with time 
groups2=df.groupby(['assignment','date'])  #group by both assignment and date  
groups2.groups    

groups2['kwh'].mean()

# if change the order 
groups2=df.groupby(['date','assignment'])  #group by both date and assignment --> group different   
groups2.groups    

#UNSTACK 

gp_mean=groups2['kwh'].mean()
gp_unstack=gp_mean.unstack('assignment') #get a dataframe of assignment, date, and the kwh how means 

gp_unstack['T']