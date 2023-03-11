#Panda Introduction and Topics
#-----------------------------
#%library

import pandas as pd

dir("D:\ML-Lab\CBAP_13DEC")
print(dir(pd), end ='  ,')

len(dir(pd))
#initial

#help
#pd.read_csv?


#pip install pydataset



#read from dataset

from pydataset import data

data('')

mtcars = data('mtcars')

mtcars

mtcars.head(5)


#write to csv

mtcars.to_csv('D:\\airline.txt')



#read a csv/ excel

import pandas as pd

#df

df = pd.read_csv('D:\\airline.txt')

df.shape

df




#pandas Series
s = pd.Series(range(1, df.shape[0]+1))
s

ps2=df.set_index(s)
ps2
#should be able to read from CSV/ only data table

#create Panda Series

#create Pandas DF

string1 = "student"
string1

for i in range(65, 100):
    print(string1 + str(chr(i)))

#You can get this error if you have variable str and trying to call str() function.
list1 = [1,2,3,4,5]
list2=['Student']

import pandas as pd
print(pd.__version__)

#panda series - single column values
ps1= pd.Series([1,4,8,11,65], dtype='int32')

#creates row index also
ps1

ps1.values

ps1.index

ps1[0]

ps1[1:4]

#assign new index when creating data
ps2=pd.Series([1,34,64,23,35], index=['a','b','c','d','e'])
ps2

#duplicate indexing
ps3=pd.Series([1,34,64,23,35], index=['a','b','c','d','d'])
ps3['d']

#pd.Series([1,34,64,23,35], index=[1,7,9,11])#error

pd.Series([1,34,64,23,35], index=[1,7,9,11,12])

ps2['c']
ps3['d']  #two values
ps2['a':'e']
#another way

ps4 = pd.Series({'a':1,'b':34,'c':64,'d':23,'e':35})
ps4

#indexing in Series
ps4[0]
ps4['a']

ps4[1:4]
ps4['a':'c']

ps4.loc['a']  #explicit index
ps4.iloc[1:3]  #implicit index 0,1,..

ps4[['a','c']]  #fancy indexing (selected)

ps4

ps4[ps4 > 30]

ps4[(ps4 > 30) & (ps4 < 50)]

ps4.keys

'a' in ps4   #index values check

ps4['a'] = 99  #change / mutate

ps4

#%%

import pandas as pd
#pandas data frame - multi column
#first create two series

course = pd.Series(['BTech','MTech','BBA','MBA'])
strength = pd.Series([100, 50, 200, 75])
fees = pd.Series([2.5, 3, 2, 4])
course, strength, fees

pd1 = pd.DataFrame([course, strength,fees])

pd1  #not the correct method of DF

pd2 = pd.DataFrame({'course':course, 'strength':strength, 'fees':fees})

d1={'course':course, 'strength':strength, 'fees':fees}

d1['course']

#count ?



pd2 #better way

pd2.index

pd2.columns

pd2.values


pd2['course']

pd2.keys


pd2[1:2] #row

pd2[1:2] #row

#indexeers - loc, iloc, ix

pd2

pd2.count()

pd2.course

pd2.columns

pd2.strength

pd2.fees

pd2 is pd2

pd2.course is pd2.fees

pd2.course is pd2['course']  #same method of access check

pd2.loc[1:3]

pd2.iloc[1:2]  #no explicit here

pd2
pd2.course=='MBA'


pd2.loc[pd2.course=='BBA']

pd2.loc[pd2.course=='MBA', 'course':'fees']


pd2.loc[pd2.course=='MBA', ['course','fees']]


pd2
pd2.iloc[1,2]


pd2.iloc[1,2] = 5.5
pd2



pd3=pd2

pd3.course is pd2.course

pd2[pd2.fees >= 20000]



#index alignment - Pd 117 - do later



#%%
#Operating on data
import numpy as np
import pandas as pd

arr = np.random.randint(0,10,4)

ser = pd.Series(arr)
ser

#df
arr1 = np.random.randint(0,10,(3,4))
arr1
df = pd.DataFrame(arr1)
df


import numpy as np
rng = np.random.randint(0,10,(3,4))
#arr2
#df = pd.DataFrame(arr2, columns=['A','B','C','D'],  index=['a','b','c'])
#df



#missing data

import pandas as pd
import numpy as np

placed = pd.Series([None,np.nan, 100, None])

placed

np.sum(placed)  #working


course = pd.Series(['BTech','MTech','BBA','MBA'])


strength = pd.Series([100, 50, 200, 75])

fees = pd.Series([2.5, 3, 2, 4])



pd3 = pd.DataFrame({'course':course, 'strength':strength, 'fees':fees, 'placed':placed})
pd3

pd3.index

pd3['course']=='MTech'

pd3[pd3['course']=='MTech']

pd3[pd3['course']=='MTech'].index

pd3.drop(1)

pd3.drop(pd3[pd3['course']=='BTech'].index)


pd3.placed.sum()

pd3.strength.sum()

pd3.placed.max()

pd3.strength.max()

placed

placed.isnull()

placed.notnull()

p2=placed.dropna()
p2

placed

import pandas as pd
#new dataframe from list values
pd4 = pd.DataFrame([['dhiraj', 50, 'M', 10000, None], [None, None, None, None, None], ['kanika', 28, None, 5000, None], ['tanvi', 20, 'F', None, None], ['poonam',45,'F',None,None],['upen',None,'M',None, None]])
pd4

pd4.dropna()  #row with any an NA value / completecases

pd4.dropna(axis='columns')  #only those columns with NA

pd4.dropna(axis=1)

pd4.dropna(axis='rows')  #rows

pd4.dropna(axis='rows', how = 'all')

pd4.dropna(axis='columns', how='all')  #all None columns

pd4.dropna(axis='columns', how='any')  #all None columns

pd4.dropna(axis='columns', thresh=3)  #min non NA values in columns

pd4.dropna(axis='rows', thresh=3)  #min min NA values in rows



#filling na values
placed= pd.Series([1,2, None, 5, None, None, 8])
placed


placed.fillna(0)

placed
placed.fillna(method='ffill')
placed.fillna(method='bfill')


#DF fill
pd4
pd4.fillna(method='ffill', axis=1)
pd4.fillna(method='ffill', axis=0)


#MultiIndex  - Later
#pag 130

x= [1,2,3]
y= [4,5,6]
z= [7,8,9]
x,y,z

np.concatenate([x,y,z])
np.concatenate([x,y,z], axis=0) #they are single dim arrays
#np.concatenate([x,y,z], axis=1)  #will not work

x=[[1,2,3],[4,5,6]]
y=[[10,11,12],[13,14,15]]
x
y

np.concatenate([x,y], axis=1) #cbind
np.concatenate([x,y], axis=0)  #rbind

#DF
grades1 = pd.DataFrame(['A','B1'])

grades1

grades1 = {'subject1': ['A1','B1','A2','A3'],'subject2': ['A2','A1','B2','B3']   }
grades1
df1 = pd.DataFrame(grades1)
df1

grades2 = {'subject3': ['A1','B1','A2','A3'],'subject4': ['A2','A1','B2','B3']}

df2 = pd.DataFrame(grades2)
df2


#join
pd.concat([df1,df2])

pd.concat([df1,df2], axis=0)

pd.concat([df1,df2], axis=1)

pd.concat([df1,df2], ignore_index=True)  #index values in order

pd =pd.concat([df1,df2], keys=['x','y'])  #adding multiple index

pd.index


print (df)


'''
Create a MultiIndex:

mi = pd.MultiIndex.from_arrays((list('abc'), list('def')))
mi.names = ['level_1', 'level_2']
Get level values by supplying level as either integer or name:

mi.get_level_values(0)
Index(['a', 'b', 'c'], dtype='object', name='level_1')
mi.get_level_values('level_2')
Index(['d', 'e', 'f'], dtype='object', name='level_2')

'''


(

import pandas as pd
#Join

rollno = pd.Series(range(1,11))

rollno

[ "Student" + str(i) for i in range(1,11)]

list(range(1,11))


name = pd.Series(["student" + str(i) for i in range(1,11)])
name

genderlist  = ['M','F']

import random

gender = random.choices(genderlist, k=10)
gender

random.choices(population=genderlist,weights=[0.4, 0.6],k=10)

import numpy as np
#numpy.random.choice(items, trials, p=probs)
np.random.choice(a=genderlist, size=10, p=[.2,.8])


import numpy as np
marks1 = np.random.randint(40,100,size=10)
marks1


pd5 = pd.DataFrame({'rollno':rollno, 'name':name, 'gender':gender, 'marks1':marks1})

pd5

#course = random.choices( population=['BBA','MBA','BTECH'] ,weights=[0.4, 0.3,0.3],k=10)
course = np.random.choice(a=['BBA','MBA','BTECH'], size=10)
course

marks2 = np.random.randint(40,100,size=10)

marks2

pd6 = pd.DataFrame({'rollno':rollno, 'course':course, 'marks2':marks2})
pd6
pd5



fees = pd.DataFrame({'course':['BBA','MBA','BTECH', 'MTECH'], 'fees':[100000, 200000, 150000, 220000]})

fees





#
pd5
pd6

#1 to 1
pd7=pd.merge(pd5, pd6)


pd7
pd7.head(2)


import pandas as pd

S1=pd.Series(range(1,11), dtype='int')
S1
name = pd.Series(["student" + str(i) for i in range(1,11)])
name

dftest1= pd.DataFrame({'rollno':S1, 'Name':name})
dftest1


S2=pd.Series(range(4,14))
# -*- coding: utf-8 -*-

pd.set_option('display.max_columns',15)
#others - max_rows, width, precision, height, date_dayfirst, date_yearfirst

import pandas as pd
import numpy as np

#read data

df = pd.read_csv('20denco.csv')
#see properties of data
df.head()

df.columns

len(df)

df.dtypes

df['partnum']=(df['partnum']).astype('str')



df.dtypes

df1 = df.describe()

df1

df.shape
df.dtypes

df.head()

'''
df['region'].unique()
df['region'] = df['region'].astype('category')
df['region'].unique()
'''

df.head()

df.region.value_counts()

df.region.value_counts().plot(kind='bar')

#do summary actions
#Who are the most loyal Customers - Improve repeated sales, Target customers with low sales Volumes
#count by customers, sort, pick top 5 

df.columns

#numpy method of counts
df.custname.value_counts()
df.custname.value_counts().sort_values(ascending=False).head(5)


#pandas way
df.groupby('custname').size()
df.groupby('custname').size().sort_values(ascending=False)
#pickup first 5
df.groupby('custname').size().sort_values(ascending=False).head(5)
#these are most loyal customers


#extra- ??
'''
df.groupby(['custname'])['margin'].nlargest(3)


df.sort_values(['revenue'], ascending= True).groupby( 'region' ).mean()
'''

#Which customers contribute the most to their revenue - How do I 
#retain these customers & target incentives
#find total revenue of each customer, sort in descending.

df.groupby('custname').aggregate({'revenue':np.sum})
df.groupby('custname').aggregate({'revenue':np.sum}).sort_values(by='revenue', ascending=False)
df.groupby('custname').aggregate({'revenue':np.sum}).sort_values(by='revenue', ascending=False).head(5)

#these are top 5 customers who gave revnue to Denco

#extras ???

'''
df.groupby('custname').aggregate({'revenue':[np.sum, 'size'] }).sort_values(by='revenue', ascending=False) #error

df[['revenue','custname']].groupby('custname').agg['size'] 
.sort_values(by='revenue', ascending=False) #error
'''




#What part numbers bring in to significant portion of revenue - 
#Maximise revenue from high value parts

#grouby partnum, find value of revenue, sell these more

df[['partnum','revenue']].sort_values(by='revenue', ascending=False)
df[['partnum','revenue']].sort_values(by='revenue', ascending=False).head(5)



#these are top parts which give max revenue
#if total sales has to be considered
df[['partnum','revenue']].groupby('partnum').sum()
df[['partnum','revenue']].groupby('partnum').sum().sort_values(by='revenue', ascending=False).head(5)
#this total revenue value giving items

gdf = df[['partnum','revenue']].groupby(['partnum'], as_index=False).agg([sum, np.mean])

gdf.to_csv("gdf.csv", index=False)

gdf = pd.read_csv("gdf.csv")
gdf

gdf=gdf[2:]
gdf = gdf.rename(columns = {'Unnamed: 0':'Partnum', 'revenue': 'revenue_Sum', 'revenue.1': 'revenue_mean'}).reindex()
gdf


gdf.sort_values(by='revenue_mean', ascending=False).head(5)
df.dtypes



#What parts have the highest profit margin - What parts are driving profits & 
#what parts need to build further
#check for margin value, their individual margin and total sales margin like revenue

df[['partnum','margin']].sort_values(by='margin', ascending=False)
df[['partnum','margin']].sort_values(by='margin', ascending=False).head(5)
df[['partnum','margin']].sort_values(by='margin', ascending=False).tail(5)

#these are top parts which give max margins
#if total sales has to be considered


pd.set_option('display.max_columns',1000)


df[['partnum','margin']].groupby('partnum').sum()
df1= df[['partnum','margin']].groupby('partnum').sum().sort_values(by='margin', ascending=False).head(5)

df1

df1.to_csv('aa.csv')



#this total revenue value giving items


#Most sold items

df.groupby('partnum').size().sort_values(ascending=False).head(5)


#which regions gave max revenue
df2 = df[['revenue', 'region']].groupby( 'region').sum().sort_values(by='revenue', ascending=False).head(5)
df2



#East gave max revenue
df[['revenue', 'region']].groupby( 'region').sum().sort_values(by='revenue', ascending=False).plot(kind='barh')

S2
name = pd.Series(["student" + str(i) for i in range(20,30)])
name

dftest2= pd.DataFrame({'rollno':S2, 'Name':name})
dftest2
dftest1


pd.merge(dftest1, dftest2, on='rollno')

pd5
pd6

pd8=pd.merge(pd5, pd6, on='rollno')

pd8

pd8.head(2)

#many to 1

pd6

fees

pd.merge(pd6, fees, on = 'course')



#
pd6b = pd.DataFrame({'rollno1':rollno, 'course':course, 'marks3':marks2})
pd6b
pd5

pdmerge= pd.merge(pd5, pd6b, left_on='rollno', right_on='rollno1')

pdmerge

pdmerge.drop('rollno1', axis=1)




rollno1 = pd.Series(range(4, 14))

pd6b = pd.DataFrame({'rollno1':rollno1, 'course':course, 'marks3':marks2})
pd6b
pd5

pdmerge= pd.merge(pd5, pd6b, left_on='rollno', right_on='rollno1')

pdmerge

pdmerge.drop('rollno1', axis=1)








pd.merge(pd5, pd6b, left_on='rollno', right_on='rollno1').drop('rollno1', axis=1).head(2)  #drop redundant coln

pd5
#with indices defined
pd5a = pd5.set_index('rollno')


pd5a


pd6
pd6a = pd6.set_index('rollno')
pd6a
pd5a

pd.merge(pd5a, pd6a, left_index=True, right_index=True)

pd.merge(pd5a, pd6, left_index=True, right_on='rollno')

pd5a
pd6a

#pd6a.join(pd5a)

pd5a
pd6

pd.merge(pd5a, pd6, left_index=True, right_on='rollno')

#set arithmetic
pd5
pd6

pd.merge(pd5, pd6, how='inner')

fees
pd6

pd.merge(pd6, fees, how='inner')


#two joins 
#left - combining on the basis of left df
#right -combining on the basis of right df

pd6
fees

pd.merge(pd6, fees, how='outer')

pd6
fees

pd.merge(pd6, fees, how='left')


pd.merge(fees, pd6, how='right')

pd5
pd6

pd4



pd4.columns

pd4.columns = ['name', 'age','gender','fees']
pd4

pd4.apply(pd.value_counts)

pd4.age

#select all from df where age>20
pd4.age > 20

pd4[pd4.age > 20]


#find out if any value in df.age is null then results True

pd4.age.isnull().any()

pd4
pd4.isnull()

pd4.isnull().any()

pd4.head(3)

pd4
pd4.dropna()

pd4
pd4.fillna(method='ffill')

pd4.fillna(method='bfill')



#pd4.dropna(inplace=True) #???? carefull it will make changes to original


pd4


pd4.columns

pd4.columns = ['name', 'age','gender','fees', 'NIL']

pd4


pd4.sort_values(ascending=True, by='name')

pd4.sort_values(ascending=False, by='name')




pd4 = pd.DataFrame([['dhiraj', 50, 'M', 10000], ['kanika', 28, None, 5000], ['tanvi', 20, 'F', None], ['upen',None,'M',None]])
pd4

pd4.columns = ['name', 'age','gender','fees']

pd4_c =pd4.copy()


pd4
pd4a= pd4.sort_values(ascending=True, by='name', inplace = True)
pd4a

pd4b=pd4.sort_values(ascending=False, by='name', inplace=False)
pd4b

pd4b=pd4.sort_values(ascending=False, by='name',inplace=True)
pd4b #change original DF



pd4
pd55= pd4.dropna(axis='index', how='all', inplace=False)
pd55




pd4

pd4.sort_values(ascending=False, by=['age'])


pd4.sort_values(ascending=False, by=['fees'], na_position='first')

pd4.sort_values(ascending=True, by=['gender','age'])


pd4.sort_values(ascending=[True,False], by=['gender','age'])
#pd4.sort() #depreciated

pd4.shape

#%%
#summary on DF: create fairly large data
import pandas as pd
import numpy as np

rollno = pd.Series(range(1,1001))

rollno

name = pd.Series(["student" + str(i) for i in range(1,1001)])
name

genderlist  = ['M','F']

import random
#gender = random.choices(genderlist, k=1000)
gender= np.random.choice(a=genderlist, size=1000,replace=True, p=[.6,.4])
gender


import collections

collections.Counter(gender)

marks1 = np.random.randint(40,100,size=1000)

marks2 = np.random.randint(40,100,size=1000)

fees = np.random.randint(50000,100000,size=1000)

fees.mean()

course = np.random.choice(a=['BBA','MBA','BTECH', 'MTech'], size=1000, p=[0.4, 0.5,0.09,0.01])

course
collections.Counter(course)

city = np.random.choice(a=['Delhi', 'Gurugram','Noida','Faridabad'], size=1000, replace=True, p=[.4,.2,.2,.2])

collections.Counter(city)


course = np.random.choice(a=['BBA','MBA','BTECH', 'MTech'], size=1000, p=[0.4, 0.5,0.09,0.01])
pd8 = pd.DataFrame({'rollno':rollno, 'name':name, 'course':course, 'gender':gender, 'marks1':marks1,'marks2':marks2, 'fees':fees,'city':city})
pd8

pd8.head(1)

pd8.shape
#start

pd8.head()

pd8.tail()


pd8.describe()

pd8.count()

pd8['gender'].value_counts()  #if col has spaces


pd8.gender.value_counts()






pd8.apply(pd.value_counts)




#pd8.apply(pd.value_counts).fillna(0)

pd8

pdg=pd8.groupby('course')
pdg

pd8.groupby('course').size()

pd8.groupby('course').count()

#conver cat columns

pd8.columns

categ = ['course', 'gender','city']


pd8.head(2)

pd9 = pd8[categ]


pd9.head()


pd9.describe()


pd8.groupby(['city','course', 'gender']).size()

pd8.groupby(['city','course', 'gender']).count()

pd8.pivot_table(index=['city','course'], columns='gender', aggfunc='size')

pd.crosstab([pd8.city, pd8.course], pd8.gender)




pd8.columns

pd8=pd8.fillna(0)
pd8.head(1)

pd8.groupby(['city','course'])['course',''].size()


course = np.random.choice(a=['BBA','MBA','BTECH', 'MTech'], size=1000, p=[0.4, 0.5,0.097,0.003])
pd8 = pd.DataFrame({'rollno':rollno, 'name':name, 'course':course, 'gender':gender, 'marks1':marks1,'marks2':marks2, 'fees':fees,'city':city})
pd8.groupby(['city','course'])['course'].size()

#pd8.groupby(['city','course'])['course'].size().fillna(0).astype(float)


a= pd.Series(['A', 'B', None, None, 'A', 'A'])

b = pd.Series([11,22,33,44])


pd8t= pd.DataFrame({"BB":b,"AA":a})

pd8t



pd8t.groupby(['AA', 'BB'])['BB'].size().fillna(0).astype(float)





#aggregate, fileter, transform 

pd8.columns

pd8

pd8.marks1.max()

pd8['marks1'].min()

pd8.groupby(['course']).size()


pd88 = pd8.groupby(['course']).get_group('MTech')

pd88

#pd8.groupby('course').aggregate(min, max)



pd8.groupby(['course']).count()


pd8.groupby('course').agg({"marks1": "sum"})

pd8.groupby('course', as_index=False).agg({"marks1": "sum"})

pd8.groupby('course', as_index=True).agg({"marks1": "sum"}) #makes course as index

pd8.groupby(['course','gender']).agg({"marks1": [min, max, np.mean], "marks2": [min, max, np.median, 'first'], "gender": ['count']})  #np.mean - numpy object

pd8


def gthan(x): 
    return x > 70


pd8['marks1'].apply(gthan).value_counts()

pd88=pd8['marks1'].apply(gthan).value_counts()

pd88.plot()

pd88.plot(kind='bar', stacked=False, figsize=[16,6], colormap='winter')


pd88
#copy to clipboard

pd88.to_clipboard(sep='  ')






'''
,rollno,name,gender,marks1,course,marks2
0,1,student1,F,73,BTECH,85
1,2,student2,F,73,BBA,52
2,3,student3,M,51,MBA,47
3,4,student4,F,94,BBA,43
4,5,student5,M,51,BTECH,45
5,6,student6,M,49,MBA,79
6,7,student7,F,97,BBA,48
7,8,student8,F,80,BTECH,79
8,9,student9,F,64,BTECH,96
9,10,student10,F,72,BBA,76
'''


pd8.to_clipboard(sep=',', index=False)

'''
rollno,name,gender,marks1,course,marks2
1,student1,F,73,BTECH,85
2,student2,F,73,BBA,52
3,student3,M,51,MBA,47
4,student4,F,94,BBA,43
5,student5,M,51,BTECH,45
6,student6,M,49,MBA,79
7,student7,F,97,BBA,48
8,student8,F,80,BTECH,79
9,student9,F,64,BTECH,96
10,student10,F,72,BBA,76
'''

pd8b=pd.read_clipboard(sep=',')
pd8b
pd8b.columns



pd8

pd8.to_csv(index=False, path_or_buf = 'pd8.csv')

pd8.to_excel("pd8.xlsx")

pd8.to_excel("pd81.xlsx",sheet_name='pd8', index=False)


#write to more than one sheet in the workbook, it is necessary to specify an ExcelWriter object:
with pd.ExcelWriter('pd8b.xlsx') as writer:
    pd8.to_excel(writer, sheet_name='first', index=False)
    pd8.to_excel(writer, sheet_name='second')
    
    

f1 = open("aa.txt", 'w')
f1 = open("aa.txt", 'r')

    
pd8.to_excel('pd8b.xlsx', engine='xlsxwriter')


pd8



def filter_func(x): 
    return x['marks1']>75

filter_func(pd8)

def filter_func(x): 
    return x>75

filter_func(pd8['marks1'])



filter_func(pd7)

pd8[pd8['marks1']>75]




pd8.groupby('course').mean()
#pd8.groupby('course').transform(lambda x: x - x.mean())
pd8.groupby(list(pd8.select_dtypes(exclude=[np.number]))).agg(np.median).reset_index()
pd8.groupby('course').agg({'marks1':np.median,'marks2':np.median,'gender':'first','fees':'last'}).reset_index()



#pivot table
import numpy as np
import pandas as pd

titanic= pd.read_csv('titanic.csv')
titanic.groupby('gender')
titanic.groupby('gender')['survived'].sum()
titanic.groupby('gender')[['survived']].sum()
titanic.groupby(['gender','class'])[['survived']].sum()
titanic.groupby(['gender','class'])[['survived']].sum().unstack()

titanic.pivot_table('survived', index='gender', columns='class', aggfunc='sum')
