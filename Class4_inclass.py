from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np 
import os

main_dir="/Users/newlife-cassie/Desktop/data"
csv1="small_data_w_missing_duplicated.csv"
csv2="sample_assignments.csv"

#IMPORTING DATA-----------------------------
df1 = pd.read_csv(os.path.join(main_dir, csv1), na_values = ['-', 'NA'])
df2 = pd.read_csv(os.path.join(main_dir, csv2), na_values = ['-', 'NA'])

# CLEAN DATA ----------------------------

## Clean df1
df1=df1.drop_duplicates()
df1=df1.drop_duplicates(['panid', 'date'], take_last = True)

## Clean df2 
df2[[0,1]] #only need the first two columns of this dataframe 
df2=df2[[0,1]] #reassigning df2 to a subset

# COPY DATAFRAMES
df3=df2   #
df4=df2.copy() #creating a copy (alter df2 does NOT affect df4)

# REPLACING DATA-----------------------------

df2.group.replace(['T','C'],[1,0])
df2.group= df2.group.replace(['T','C'],[1,0])

# df3 changed as we changed df2, even though we did not change df3
# df4 is an original copy of df2 

# MERGING-------------------------------------

pd.merge(df1,df2) #attaching df2 to df1, default: many to one merge, use an
#intersection. automatically finds the keys they have in common 

pd.merge(df1,df2,on=['panid']) #specify the key to merge on 

pd.merge(df1,df2, on=['panid'],how='inner') #merging for intersection of the keys
                                            #merged set only have panid1-4
pd.merge(df1,df2, on=['panid'],how='outer') #merging all together 
                                            # merged set have panid 1-5(which df2 had)
df5= pd.merge(df1,df2,on=['panid']) #assigning the merged object to df5 

# ROW BINDS and COLUMN BINDS 

## row bind 
pd.concat([df2,df4]) #the default is to row bind 
pd.concat([df2,df4], axis=0) #same as above 
pd.concat([df2,df4], axis=0,ignore_index=True) #index change from (1-5)&(1-5) to (1-10)

pd.concat([df2,df4], axis=1) #column binding 

