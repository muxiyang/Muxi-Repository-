from pandas import Series, DataFrame
import pandas as pd
import numpy as np 
import os

# DUPLICATED VALUES
## creat new dataframe 

zip3=zip(['red','green','blue','organge']*3,[5,10,20,40]*3,[';(',':D',':D']*4)

df3=DataFrame(zip3, columns=['A','B','C'])

## pandas method 'duplicated'
df3.duplicated() #searching from top to bottom by default 
df3.duplicated (take_last=True) # searches bottom to top 

## subset duplicated values 
df3.duplicated(subset=['A','B']) #--> compare colum A and B only
df3.duplicated(['A','B']) #--> don't have to write subset, same as before

## HOW to get all values that have duplicates 
t_b=df3.duplicated()
b_t=df3.duplicated(take_last=True)
unique=~(t_b|b_t) #true from top to bottom or bottom to top returns true 
unique=~t_b & ~b_t

df3[unique] #values that are unique 
df3[t_b] #from bottom to top, these are the things that are duplicated

#DROPPING DUPLICATES-------------------------------------------------
df3.drop_duplicates () #drop from top to bottom 
df3.drop_duplicates(take_last=True) # drop from bottom up 

##this is the same as 
t_b=df3.duplicated()
df3[~t_b]  #df3 is values that are not duplicated (~ NOT)
df3.drop_duplicates ()==df3[~t_b]

##subset criteria
df3.drop_duplicates(['A','B'])

# WHEN TO USE -------------------------------------------------------
## if you want to keep the first duplicated value (from top) and remove others
df3.drop_duplicates()

## same, but from the bottom 
df3.drop_duplicates(take_last=True)
 
## purge all values that are duplicates
t_b=df3.duplicated()
b_t=df3.duplicated(take_last=True) #complement where either is true 
unique=~(t_b|b_t)