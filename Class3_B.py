from pandas import Series, DataFrame 
import pandas as pd 
import numpy as np 
import os 

main_dir="/Users/newlife-cassie/Desktop/Data"
git_dir="/Users/newlife-cassie/Desktop/PUBPOL590"
csv_file="sample_data_clean.csv"

# FOR LOOPS -----------------------------------
df= pd.read_csv(os.path.join(main_dir,csv_file))

list1=range(10,15)
list2=['a','b','c']
list3=[1,'a',True]

## iterating over elements (for loops)
for v in list1: #for v:go through each object in the list and execute the command on each of them
    v
    
for v in list1: 
    print(v)   #print displays things in the loop 
    
for v in list2: 
    print(v)   
    
for v in list3:
    print(v,type(v)) 

## poulating lists    
list1 # all int 
list4 = [] #empty list

for v in list1:
    v2 = v**2   #v**: v squared 
    list4.extend([v2])        # extend only accept one object, use []
    
list5=[]
for v in list1:
    v2=v**2 
    list5.append(v2)        #appends whatever object as is--> multiple objects 

[v**2 for v in list1]        #[]; outcome is a list,look for each object and squared and loop

list6= [v**2<144 for v in list1] #true or false, v squared is less than 144

## iterating using enumerate 
list7= [[i,v/2] for i, v in enumerate(list1)]
list8=[[i,float(v)/2] for i, v in enumerate(list1)]

# ITERATE THROUGH A SERIES----------------------------------
s1=df['consump']
[v>2 for v in s1]

[[i,float(v)*0.3] for i, v in s1. iteritems()]

# ITERATE THROUGH a DATAFRAME--------------------------------
[v for v in df]
[df[v] for v in df]
[[i,v] for i, v in df.iteritems()]