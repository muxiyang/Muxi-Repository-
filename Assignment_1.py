from pandas import Series, DataFrame
import pandas as pd
import numpy as np 

##IMPORTING DATA
#assigning file path
main_dir="/Users/newlife-cassie/Desktop/PUBPOL590/Assignment_1"
txt_file="/File1_small.txt"

main_dir+txt_file

#read table
df=pd.read_table(main_dir+txt_file,sep=" ")

#check obj type 
type(df)
list(df)

# row slicing
df[60:100]

#extraction by boolean indexing
c=df.kwh
df[df.kwh>30]