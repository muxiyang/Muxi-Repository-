from pandas import Series, DataFrame 
import pandas as pd 
import numpy as np 
import os 

main_dir="/Users/newlife-cassie/Desktop/Data"
git_dir="/Users/newlife-cassie/Desktop/PUBPOL590"
csv_file_good="sample_data_clean.csv"
csv_file_bad="/sample_data_clean.csv"

# OS MODULE----------------------
df=pd.read_csv(os.path.join(main_dir,csv_file_bad))

df=pd.read_csv(os.path.join(main_dir,csv_file_good))

## strings 
str1="hello,computer"
str2="hello,human"
str3=u'eep'   #uni-code: unversal across any platform 

type(str1) # type str 
type(str2) # type str 
type(str3) # type unicode

##numeric 
int1=10
float1=22.2 #anything that has a decimal 
long1=4.2589780899999999999 #anything that is a enormous long number 

##logical 
bool1= True #bool: True or False 
notbool1=0 #False
bool2=bool(notbool1) #tansfer between True/False and integers 

# CREATING LISTS AND TUPLES 

## in brief, lists can be changed, tuples cannot 
## we almost exclusively use lists 

list1=[]
list1
list2=[1,2,3] 
list2[2] #an element in the list, elements start from 0, so element to equals 3
list2[2]=5

##tuples, cant change 
tup1=(8,3,19)
tup1[2] 
tup1[2]=5 #output can't change 

## convert 
list2=list(tup1)
tup2=tuple(list1)

##list can be append and extended 
list2.append([3,90]) 
##everything in the square bracket is the object, which is going to be append
##added [3,90] to list2 
len(list2) # length of list2 should be 4, [1,2,5,[]]

list3=[8,3,90]
list3.extend([6,88]) ##add pure numbers to the list 

len(list3) #length of list3 should be 5, [8,3,90,6,88]



#CONVERTING LISTS TO SERIES AND DATAFRAME

list4= range(100,105) # range (n,m) gives a list from #n to m-1

list4 # a series of numbers starting from 100 to 104
list5 = range(5)# range(n) gives list from 0 to m-1
list6=['q','r','s','t','u'] 

## list to series 
s1=Series(list4)  ## turn a list (horizontal) into a column with index
s2=Series(list6)

## create DataFrame from lists OR series 
zip(list4,list6)
list7= range(60,65)
zip(list4,list6,list7)

zip1=zip(list4,list6,list7)
df1=DataFrame(zip1)
#extract the first column--> df1[0]

df2=DataFrame(zip1,columns=['two','apple',':)'])
#extract the column "two"--> df2['two']

df3= DataFrame(zip1,columns=[2,'2',':)'])
#summary: extract the column, df3[what ever in the line above (include all '')]
#the 2,'2' ';)' are called keys. they can be numbers or any notation

df3[2] # reference column with key (str) '2'
df3[3:4] # slice out row 3 

df3[['2',':)']][3:4] #get only column '2' and smile face & then get row 3 

## make dataframe using dict notation 
df4 = DataFrame ({':(' : list4, 9: list6}) 
#':(" is the key for list4 and 9 is the key for list 6
# the order of the columns is determined by the nature of the key 
# numbers comes before letters, before notations 

dict1={':(' : list4, 9: list6}

## Stack two DataFrame
s1=Series(list4)
s2=Series(list6)
df5= pd.concat([s1,s2])



