from pandas import Series, DataFrame
import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt

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
df=pd.concat([pd.read_csv(v, names=['panid','date','kwh'], parse_dates=[1],header=None) for v in paths],ignore_index=True)
## dealing with date: parse_dates=column number --> convert into date format 
#Date data is often strings, which don't have value and can't be sort in order 

df.sort(['panid','date'])
#sort according to panid and date 

#Convert into date format 

# IMPORT and MERGE
df_assign=pd.read_csv(root+"sample_assignments.csv",usecols=[0,1])
df=pd.merge(df,df_assign)

#GROUPBY aka "split,apply,combine"--------------------------------------------------

## see more at the website 

#split by c/T, pooled without time 

#splitting by assignment 
groups1=df.groupby(['assignment'])  #just group by assignment, don't care about time, same as collapse in STATA
groups1.groups    # result is a index (dictionary) of treatment and control group separate
#don't do group1.groups with big data, it will crash 

gd1=groups1.groups #define a dictionary 

#Peak inside dictionary 
gd1.keys() #the C and T keys, keys refers to the grouping method 
gd1.values ()  # this is a list of index , can use numerical index 
gd1.values()[0] #0 is C, 1 is T
gd1['C'] #gd1 is a dictionary, so must use keys to get data 
gd1.viewvalues() #See all possible values of gd1 dictionary 

# iteration properties of a dictionary 
[ v for v in gd1.itervalues()] 
gd1.values()   # these are equivalent 

[k for k in gd1.iterkeys()]
gd1.keys()    #these are equivalent 

[(k,v) for k,v in gd1. iteritems()] # tubule of a key (c or T) and a list of values following 
#the key 
gd1           # these are equivalent 


#apply the mean 
groups1['kwh'].apply(np.mean)   #cannot see the things in the groups1, just calaculate the mean 
#the result is a cross-sectional (no time indicator) mean 

groups1['kwh'].mean() #internal function, faster --> use this!!!

%timeit -n 100 groups1['kwh'].apply(np.mean) #time how long it takes to calculate 
%timeit -n 100 groups1['kwh'].mean() #time how long it takes to calculate 
#the internal function is faster 

#Split by control and treatment but pooling with time 
groups2=df.groupby(['assignment','date'])  #group by both assignment and date  
gd2=groups2.groups 
gd2 #key is both assignment and date, the [] shows the row values in each key 
gd2.keys()  
groups2.mean() 

groups2['kwh'].mean() # group by row index, and calculate the mean over kwh for items assigned 
#to a particular group (for example everything in group c and on Jan1). 

# Split and apply (Panel/ time series data) 
groups2=df.groupby(['date','assignment'])  #group by both date and assignment --> group different   
groups2.groups    

#UNSTACK 

gp_mean=groups2['kwh'].mean()
gp_unstack=gp_mean.unstack('assignment') #get a dataframe of assignment, date, and the kwh how means 

gp_unstack['T']

# TESTING FOR BALANCE (OVER_TIME)----------------------------------------------
from scipy.stats import ttest_ind 
from scipy.special import stdtr

## example using t-test
a=[1,4,9,2]
b=[1,7,8,9] 

t,p=ttest_ind(a,b, equal_var= False)
t
p
#(array(-0.8896802485305646), 0.40789376238402786) First is t-statistics, then p-value 

# set up data
grp=df.groupby(['assignment','date'])

#get separate sets of treatment by date

trt={k[1]:df.kwh[v].values for k,v in grp.groups.iteritems() if k[0]=='T'}
#get only the treatment data + transform row index into actual kwh values 

grp.groups.keys()[0] # The first key, which is a tapule with assignment and date
k=grp.groups.keys()[0]
k[0]
#[0] and [1] refers to the first and second item in the tapule, which is (assignment, date)

crt={k[1]:df.kwh[v].values for k,v in grp.groups.iteritems() if k[0]=='C'}
# get only the control data + transform row index into actual kwh values 

keys=trt.keys() #why doesn't it contain time, but not assignments? 

diff={k: (trt[k].mean()-crt[k].mean()) for k in keys}

# Create dataframes of this information 
tstats=DataFrame([(k,np.abs(ttest_ind(trt[k],crt[k],equal_var=False)[0])) for k in keys],columns=['date','tstats'])
pvals=DataFrame([(k,np.abs(ttest_ind(trt[k],crt[k],equal_var=False)[1])) for k in keys],columns=['date','pvals'])
t_p=pd.merge(tstats,pvals)

#sort and reset index 
t_p.sort(['date'], inplace=True) #
t_p.reset_index(inplace=True,drop=True)

# t-stats --> comparisons! 
diff={k: (trt[k].mean()-crt[k].mean()) for k in keys}
tstats={k: float(ttest_ind(trt[k],crt[k], equal_var=False)[0])for k in keys}
pvals={k: float(ttest_ind(trt[k],crt[k], equal_var=False)[1])for k in keys}
t_P={k:(tstats[k],pvals[k]) for k in keys}

#Plotting 
fig1=plt.figure()
ax1=fig1.add_subplot(2,1,1)
ax1.plot(t_p['tstats'])
ax1.axhline(2,color='r',linestyle='--')
ax1.set_title('tstats over time')

ax2=fig1.add_subplot(2,1,2)
ax2.plot(t_p['pvals'])
ax2.axhline(0.05,color='r',linestyle='--')
ax2.set_title('pvalue over time')