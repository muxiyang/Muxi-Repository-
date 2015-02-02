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
df1= pd.read_csv(os.path.join(main_dir,csv_file),na_values=missing)
df1.head(10)

#2
df1.drop_duplicates 

#3
df1.isnull()
df1['consump'].isnull()
rows=df1['consump'].isnull()
df1[rows]

df2=df1[rows]

#4
df1.duplicated(subset=['panid','date'])
t_b=df1.duplicated()
b_t=df1.duplicated(take_last=True)

df3=df1[t_b]
df4=df3. dropna (axis=0, how='any')

#5
df4['consump'].mean()



