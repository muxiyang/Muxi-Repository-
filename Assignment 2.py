from pandas import Series, DataFrame
import pandas as pd
import numpy as np 
import os

main_dir="/Users/newlife-cassie/Desktop/data"
csv_file="small_data_w_missing_duplicated.csv"

df= pd.read_csv(os.path.join(main_dir,csv_file))

#1
df.head()
df.head(10)

df.tail (10)

missing=['.','NA','NULL',' ','-']
df= pd.read_csv(os.path.join(main_dir,csv_file),na_values=missing)
df.head(10)

#2
df.duplicated()
df.drop_duplicates ()

#3
df.isnull()
df['consump'].isnull()
rows=df['consump'].isnull()
df[rows]

#4
df.duplicated(subset=['panid','date'])
t_b=df.duplicated()
b_t=df.duplicated(take_last=True)

df1=df[t_b]

df1. dropna (axis=0, how='any')

#5
df['consump'].mean()



