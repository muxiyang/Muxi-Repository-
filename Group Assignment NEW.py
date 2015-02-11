from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np 
import os

main_dir="/Users/newlife-cassie/Desktop"

# ADVANCED PATHING----------------------------------------------------
root = main_dir + "/Group_Assignment_Data/"
paths = [root + v for v in os.listdir(root) if v.startswith("File")]
paths1=[root +"File"+str(v)+".txt" for v in range (1,7)]

#IMPORT DATA-----------------------------------------------------------
missing=['.','NA','NULL',' ','0','-','999999999']
list_of_dfs = [pd.read_table(v, names = ['ID', 'date', 'kwh'], sep = " ",na_values= missing,skiprows=6000000,nrows=1500000) for v in paths]

df_assign = pd.read_csv(root + "SME and Residential allocations.csv",na_values= missing, usecols = [0,1,2,3,4])

#STACKING AND MERGING----------------------------------------------------  
df_stack = pd.concat(list_of_dfs, ignore_index = True) 
del list_of_dfs
df = pd.merge(df_stack, df_assign)
df_copy=df.copy

# MISSING DATA----------------------------------------------------------
df=df.dropna(how="all") #drop ROWS with ALL missing values

#CHECK DAYLIGHT SAVING DATES---------------------------------------------

## Extract hour counts 
df['tt']=df['date']% 100
## Extract day counts
df['dd']=(df['date']-df['date']%100)/100

## Check at which dates have hour 49 and 50, or greater than 50
df_tt1= df[df.tt==49]
df_tt2= df[df.tt==50]
df_tt3=df[df.tt>50] 

## Check the daylight saving dates mentioned in the CER FAQ
df_dd1=df[df.date==45202] 
df_dd2=df[df.date==45203]
df_dd3=df[df.date==66949]
df_dd4=df[df.date==66950]
df_dd5=df[df.date==29849]
df_dd6=df[df.date==29850]

## Delete hour 3 and 4 of day 298. Each hour after that were moved 1 hour forward
df1=df.copy()
df1=df1[df1.date!=29803]
df1=df1[df1.date!=29804]
df1_copy=df1.copy()
new_date=df1.date
new_tt=df1.tt
for i in range(5,51):
    print(i)
    ind_bin=((df1.dd==298) & (df1.tt==(i)))
    ind=new_date[ind_bin].index
    new_date[ind]=new_date[ind]-2
    new_tt[ind]=new_tt[ind]-2

# DROP DUPLICATES------------------------------------------------------------
t_b=df1.duplicated()
b_t=df1.duplicated(take_last=True)
unique=~(t_b|b_t) 
unique=~t_b & ~b_t
df2=df1[unique]
