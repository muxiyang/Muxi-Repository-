from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from dateutil import parser   # use this to ensure dates are parsed correctly
import statsmodels.api as sm 
import os 

main_dir="/Users/newlife-cassie/Desktop/PUBPOL590/Data_All/09Data"
csv_file="allocation_subsamp.csv"

df= pd.read_csv(os.path.join(main_dir,csv_file))

#1. Creating vectors--------------------------------------
c=df['ID'] [((df['stimulus'] == 'E') & (df['tariff'] == 'E'))]
A1=df['ID'] [((df['stimulus'] == '1') & (df['tariff'] == 'A'))]

A3=df['ID'] [((df['stimulus'] == '3') & (df['tariff'] == 'A'))]

B1=df['ID'] [((df['stimulus'] == '1') & (df['tariff'] == 'B'))]

B3=df['ID'] [((df['stimulus'] == '3') & (df['tariff'] == 'B'))]

#2. set random seed--------------------------------------- 
np.random.seed(seed=1789)

#3. Extract sample size-----------------------------------
c=np.random.choice(c, size=300, replace=False, p=None)
A1=np.random.choice(A1, size=150, replace=False, p=None)
A3=np.random.choice(A3, size=150, replace=False, p=None)
B1=np.random.choice(B1, size=50, replace=False, p=None)
B3=np.random.choice(B3, size=50, replace=False, p=None)

#4. Create Dataframe with all sampled IDs------------------
a1=Series(A1)
a3=Series(A3)
b1=Series(B1)
b3=Series(B3)
c=Series(c)

dfc=DataFrame(c,columns=['ID'])
dfc['vector']=1

dfa1=DataFrame(a1,columns=['ID'])
dfa1['vector']=2

dfa3=DataFrame(a3,columns=['ID'])
dfa1['vector']=3

dfb1=DataFrame(b1,columns=['ID'])
dfa1['vector']=4

dfb3=DataFrame(b3,columns=['ID'])
dfb3['vector']=5

df1=pd.concat([dfc,dfa1,dfa3,dfb1,dfb3])

#5. Import consumption data---------------------------------
csv_file1="kwh_redux_pretrail.csv"
df2= pd.read_csv(os.path.join(main_dir,csv_file1),header=0,parse_dates=[2],date_parser=np.datetime64)

# 6.Merge data----------------------------------------------
df=pd.merge(df1, df2)

# 7.Monthly aggregation-------------------------------------
df['date'] = df['date'].apply(np.datetime64)
df['year'] = df['date'].apply(lambda x: x.year)
df['month'] = df['date'].apply(lambda x: x.month)

grp=df.groupby(['year','month','ID','vector'])
df3=grp['kwh'].sum().reset_index()

#8. pivot data------------------------------------------------
df3['mo_str']=['0'+str(v) if v<10 else str(v) for v in df3['month']]
# makes sure month are in the numerical order --> if month is less than 10 then string (add an zero in front)
df3['kwh_ym']='kwh_'+df.year.apply(str)+"_"+ df3.mo_str.apply(str)

df3_piv=df3.pivot('ID','kwh_ym','kwh')
df3_piv.reset_index(inplace=True)
df3_piv.columns.name= None

# 9.Merge with treatment data---------------------------------
df=pd.merge(df, df3_piv)
del df3_piv, df3

#10. Logit Model----------------------------------------------

# GENERATE DUMMY VARIABLES FROM QUALITATIVE DATA
 
df4=pd.get_dummies(df,columns = ['vector'])
df4.drop(['vector_1'],axis=1, inplace=True)

## SET UP THE DATA FOR LOGIT 
kwh_cols=[v for v in df4.columns.values if v.startswith('kwh')]

##
cols=['gender_F']+kwh_cols
#creating a list of x values 

## SET UP Y, X 
y=df1['assignment']
X=df4[cols]
X=sm.add_constant(X)

## LOGIT----------------------
logit_model=sm.Logit(y,X)
logit_results=logit_model.fit()
print(logit_results.summary())


#Additional loops for section 
ids=df_allloc['ID']
tariffs=[v for v in pd.unique(df_alloc['tariff'])if v !='E']
stimuli=[v for v in pd.unique(df_alloc['stimulus'])if v !='E']

EE=np.random.choice(ids[df_alloc['tariff']=='E'],300,false)

for i in tariffs:
    for j in stimuli:
        n=150 if i=='A' else 50
        temp=np.random.choice(ids[(df_alloc['tarrif']==i)&(df_alloc['stimulus']==j)],n,false)
        EE=np.hstack((EE,temp))
        