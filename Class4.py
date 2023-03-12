# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:30:00 2023

@author: vikas
"""


import pandas as pd
import numpy as np

rollno = pd.Series(range(1,1001), dtype='object')
name = pd.Series(["student" + str(i) for i in range(1,1001)])

genderlist  = ['M','F']
import random
gender= np.random.choice(a=genderlist, size=1000,replace=True, p=[.6,.4])

marks1 = np.random.randint(40,100,size=1000)
marks2 = np.random.randint(40,100,size=1000)

fees = np.random.randint(50000,100000,size=1000)

course = np.random.choice(a=['BBA','MBA','BTECH', 'MTech', None], size=1000)

city = np.random.choice(a=['Delhi', 'Gurugram','Noida','Faridabad', None], size=1000, replace=True, p=[.4,.2,.2,.2])

pd1 = pd.DataFrame({'rollno':rollno, 'name':name, 'course':course, 'gender':gender, 'marks1':marks1,'marks2':marks2, 'fees':fees,'city':city})
pd1

pd1.head()
pd1.head(10)

pd1.tail()
pd1.tail(10)


pd1.dtypes
pd1.describe()


pd1.columns

pd1.corr()

pd1.isna()

import numpy as np

np.sum(pd1.isna())

pd1['course'].isna()

pd1[pd1['course'].isna()]

pd1['course'][pd1['course'].isna()] = 'NotAvaialble'

pd1['course']

pd1.to_csv('Student.csv')

#Groupby Clauses


pd1


pd2 = pd1[['course', 'gender', 'city', 'marks1', 'marks2', 'fees']]
pd2



pd2.groupby('course').size()


# Aggregation Functions eg. size(), max(), min(), average()

pd2.groupby('course').count()


pd2.groupby(['course', 'city']).size()

pd2.dtypes

pd2.groupby(['course', 'city', 'gender']).size()


pd1.groupby(['course']).aggregate(min)

pd1

ps1 = pd1.groupby(['course']).agg({'marks1':min})
ps1

ps1.index

val = ps1[ps1.index=='BBA'].values[0][0]
val

pd1[pd1['marks1']==val]


ps1 = pd1.groupby(['course']).agg({'marks1': [min, max, np.median, np.mean, np.std]})
ps1


ps1 = pd1.groupby(['course']).agg({'marks1': [min, max], 'marks2': [np.median, np.mean, np.std]})
ps1

pd2.pivot_table(index=['city'], columns = ['gender','course'], aggfunc='size')



#Merge or Join

import pandas as pd
rollno = pd.Series(range(1,11))

name = pd.Series(["student" + str(i) for i in range(1,11)])

genderlist  = ['M','F']
import numpy as np
gender = np.random.choice(a=genderlist, size=10)

marks1 = np.random.randint(40,100,size=10)
marks1

pd5 = pd.DataFrame({'rollno':rollno, 'name':name, 'gender':gender})
pd5

course = np.random.choice(a=['BBA','MBA','BTECH'], size=10)

marks2 = np.random.randint(40,100,size=10)

pd6 = pd.DataFrame({'rollno':rollno, 'course':course, 'marks1':marks1, 'marks2':marks2})

fees = pd.DataFrame({'course':['BBA','MBA','BTECH', 'MTECH'], 'fees':[100000, 200000, 150000, 220000]})

pd5
pd6
fees

pd7 = pd.merge(pd5, pd6)
pd7

rollno = pd.Series(range(6,16))
pd8 = pd.DataFrame({'rollno':rollno, 'course':course, 'marks1':marks1, 'marks2':marks2})

pd5
pd8

pd9 = pd.merge(pd5, pd8)
pd9

pd10 = pd.merge(pd5, pd8, how='outer')
pd10



pd11 = pd.merge(pd5, pd8, how='left')
pd11

pd12 = pd.merge(pd5, pd8, how='right')
pd12


name = pd.Series(["student" + str(i) for i in range(6,16)])

pd13 = pd.DataFrame({'rollno':rollno,'name':name, 'course':course, 'marks1':marks1, 'marks2':marks2})

pd14 = pd.merge(pd5, pd13,left_on=('rollno','name'), 
                right_on=('rollno','name'), how='inner')
pd14

pd5
pd13



pd14 = pd.merge(pd5, pd13,left_on=('rollno','name'), 
                right_on=('rollno','name'), how='outer')
pd14


'''
DBMS Table Keys

Single Key
roll =1 and roll =1 fetch data

Composite Keys

rollno and name

'''



pd13
fees

pd14 = pd.merge(pd13, fees)
pd14

pd1.to_excel('Data.xlsx')
pd1.to_excel("Data1.xlsx",sheet_name='Res', index=False)




pd1

pd2 = pd1.groupby('course').sum()

pd3 = pd1.groupby(['course','city']).agg({'marks1':[min,max], 'marks2':[np.median, np.mean, np.std]})

pd1
pd2
pd3

with pd.ExcelWriter('Data2.xlsx') as writer:
    pd1.to_excel(writer, sheet_name='CoreData', index=False)
    pd2.to_excel(writer, sheet_name='CourseGroups')
    pd3.to_excel(writer, sheet_name='MarksDetail')



#Denco Case Study

import pandas as pd
import numpy as np


df = pd.read_csv('20denco.csv')
df

df.columns
df.dtypes

df['partnum'] = df['partnum'].astype('object')
df.dtypes

df.describe()
df.count()

df.head()



#Find out Top Loyal Customers


df.groupby(['custname']).size()

df.groupby(['custname']).size().sort_values(ascending = False)

df.groupby(['custname']).size().sort_values(ascending = False).head()

df.groupby(['custname']).size().sort_values(ascending = False).head().plot(kind='bar')


#Find highest amount customers Revenue

df.columns

df.groupby(['custname']).agg({'revenue':sum})

df.groupby(['custname']).agg({'revenue':sum}).sort_values(ascending=False, 
                                                          by='revenue')

df.groupby(['custname']).agg({'revenue':sum}).sort_values(ascending=False, 
                                                          by='revenue').head()

df2= df.groupby(['custname']).agg({'revenue':sum}).sort_values(ascending=False, 
                                                          by='revenue').head()

df2.plot(kind='bar')


df.columns

#Partnum generates highest Margin

df.groupby(['partnum']).agg({'margin':sum}).sort_values(ascending=False, 
                                                          by='margin').head()


#which regions gave max revenue

df3 = df.groupby(['region']).agg({'margin':sum}).sort_values(ascending=False, 
                                                          by='margin').head()


df3.plot(kind='bar')



#RFM Analysis

import pandas as pd

df = pd.read_csv('OnlineRetail.csv',encoding='latin')

df.columns

df.dtypes


df = pd.read_csv('OnlineRetail.csv',encoding='latin', parse_dates= ['InvoiceDate'])

df.dtypes

df

pd.set_option('display.max_columns', None)

df.head()


df['CustomerID'] = df['CustomerID'].astype('object')

df.dtypes


df

df = df[df['CustomerID'].notnull()]

df = df.reset_index()


#Only Country 'UK'


df['Country'].value_counts().plot(kind='bar')

df = df[df['Country']=='United Kingdom'].reset_index()
df

df.columns

df['Amount'] = df['Quantity'] * df['UnitPrice']

df = df.drop(['level_0', 'index','Description', 'Country', 'Quantity',
              'UnitPrice'], axis=1)

df.dtypes

import datetime as dt

PRESENT = dt.datetime(2011,12,10)

PRESENT 

#RFM

rfm = df.groupby('CustomerID').agg({'InvoiceDate': lambda date: (PRESENT - date.max()).days,
                                    'InvoiceNo': len,
                                    'Amount':sum})

rfm.columns = ['Recency', 'Frequency', 'Monetory']
rfm


rfm['RQ'] = pd.qcut(rfm['Recency'], 3, ['1','2','3'])
rfm

rfm['FQ'] = pd.qcut(rfm['Frequency'], 3, ['3','2','1'])
rfm

rfm['MQ'] = pd.qcut(rfm['Monetory'], 3, ['3','2','1'])
rfm


rfmAnalysis = rfm['RQ'].astype('object')+rfm['FQ'].astype('object')+rfm['MQ'].astype('object')

rfmAnalysis

rfmAnalysis[rfmAnalysis=='113']




# Plots

import matplotlib.pyplot as plt

Year = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]
Unemployment_Rate = [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
Unemployment_Rate1 = [1.8,2,8,4.2,6.0,7,3.5,5.2,7.5,5.3]


plt.plot(Year,Unemployment_Rate)
plt.title('Year vs UR')
plt.xlabel('Year')
plt.ylabel('UR')



plt.plot(Year,Unemployment_Rate)
plt.plot(Year,Unemployment_Rate1)
plt.title('Year vs UR')
plt.xlabel('Year')
plt.ylabel('UR')
plt.show()



plt.plot(Year,Unemployment_Rate, label = 'UR1')
plt.plot(Year,Unemployment_Rate1, label = 'UR2')
plt.title('Year vs UR')
plt.xlabel('Year')
plt.ylabel('UR')
plt.legend()
plt.show()



plt.plot(Year,Unemployment_Rate, label = 'UR1', marker='o')
plt.plot(Year,Unemployment_Rate1, label = 'UR2', marker='*')
plt.title('Year vs UR')
plt.xlabel('Year')
plt.ylabel('UR')
plt.legend()
plt.show()


plt.plot(Year,Unemployment_Rate, label = 'UR1', marker='o', color='green')
plt.plot(Year,Unemployment_Rate1, label = 'UR2', marker='*', color='black')
plt.title('Year vs UR')
plt.xlabel('Year')
plt.ylabel('UR')
plt.legend()
plt.show()




plt.plot(Year,Unemployment_Rate, label = 'UR1', marker='o', color='y')
plt.plot(Year,Unemployment_Rate1, label = 'UR2', marker='*', color='r')
plt.title('Year vs UR')
plt.xlabel('Year')
plt.ylabel('UR')
plt.legend()
plt.show()




'''
R 0-255
B 0-255
G 0-255
00 - FF


RBG
0000FF
'''

cp = ['#B7E3CC', '#B7E3CC', '#7B4B94']


plt.plot(Year,Unemployment_Rate, label = 'UR1', marker='o', color=cp[0])
plt.plot(Year,Unemployment_Rate1, label = 'UR2', marker='*', color=cp[2])
plt.title('Year vs UR')
plt.xlabel('Year')
plt.ylabel('UR')
plt.legend()
plt.show()




plt.plot(Year,Unemployment_Rate, label = 'UR1', marker='o', color='y')
plt.plot(Year,Unemployment_Rate1, label = 'UR2', marker='*', color='r')
plt.title('Year vs UR', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('UR')
plt.legend()
plt.grid()
plt.show()


r1 = list(range(0,16))
r1

plt.plot(Year,Unemployment_Rate, label = 'UR1', marker='o', color='y')
plt.plot(Year,Unemployment_Rate1, label = 'UR2', marker='*', color='r')
plt.title('Year vs UR', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('UR')
plt.yticks(r1)
plt.legend()
plt.grid()
plt.show()



r1 = list(range(0,16))
r1

r2 = list(range(1920,2020,10))
r2
plt.plot(Year,Unemployment_Rate, label = 'UR1', marker='o', color='y')
plt.plot(Year,Unemployment_Rate1, label = 'UR2', marker='*', color='r')
plt.title('Year vs UR', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('UR')
plt.xticks(r2)
plt.yticks(r1)
plt.legend()
plt.grid()
plt.show()




#Bar Chart


plt.bar(Year,Unemployment_Rate)
plt.title('Year vs UR', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('UR')
plt.grid()
plt.show()



Country = ['USA','Canada','Germany','UK','France']
GDP_Per_Capita = [45000,42000,52000,49000,47000]


plt.bar(Country, GDP_Per_Capita)
plt.title('Country Vs GDP Per Capita')
plt.xlabel('Country')
plt.ylabel('GDP Per Capita')
plt.show();




New_Colors = ['green','blue','purple','brown','teal']

plt.bar(Country, GDP_Per_Capita, color= New_Colors)
plt.title('Country Vs GDP Per Capita')
plt.xlabel('Country')
plt.ylabel('GDP Per Capita')
plt.show();




New_Colors = ['green','blue','purple','brown','teal']

plt.barh(Country, GDP_Per_Capita, color= New_Colors)
plt.title('Country Vs GDP Per Capita')
plt.xlabel('Country')
plt.ylabel('GDP Per Capita')
plt.show();




#Scatter

import numpy as np
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

plt.scatter(x,y)



import pandas as pd

df = pd.read_csv('house.csv')
df

df.columns

plt.scatter(df['Price'], df['area'])
plt.xlabel('Price')
plt.ylabel('Area')



df = pd.read_csv('Housing.csv')

df

df = df.dropna()

df.columns
plt.scatter(df['price'], df['area'])
plt.xlabel('Price')
plt.ylabel('Area')



from pydataset import data

mt = data('mtcars')

mt.columns

plt.scatter(mt['hp'], mt['mpg'])
plt.xlabel('HP')
plt.ylabel('MPG')

mt.head(10)


plt.scatter(mt['hp'], mt['mpg'], c=mt['gear'])
plt.xlabel('HP')
plt.ylabel('MPG')



plt.scatter(mt['hp'], mt['mpg'], c=mt['gear'], s=mt['disp'])
plt.xlabel('HP')
plt.ylabel('MPG')




import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

tdf = sns.load_dataset('tips')
tdf

tdf['smoker'].dtype

tdf['smoker'] = tdf['smoker'].astype('object')

tdf['smoker'][tdf['smoker']=='No'] = 0
tdf['smoker'][tdf['smoker']=='Yes'] = 1

plt.scatter(tdf['total_bill'], tdf['tip'],c=tdf['smoker'])



#Initial Parameters Setup


import matplotlib.pyplot as plt

plt.rc('font', size=12)     
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=16)     
plt.rc('legend', fontsize=16)     
     

plt.barh(Country, GDP_Per_Capita, color= New_Colors)
plt.title('Country Vs GDP Per Capita')
plt.xlabel('Country')
plt.ylabel('GDP Per Capita')
plt.show();




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


n1 = np.random.randint(0,100, size=1000)
plt.hist(n1)

np.mean(n1)
np.median(n1)
np.std(n1)
np.min(n1)
np.max(n1)



n2= np.random.normal(50, 10, size=1000)
plt.hist(n2)

n2
np.mean(n2)
np.median(n2)
np.std(n2)
np.min(n2)
np.max(n2)


n2 = np.random.binomial(1, 0.2, size=100)
n2
plt.hist(n2)



'''
n2 = np.random.multinomial(4, [0.3,0.2, 0.2, 0.3] , size=100)
n2
plt.hist(n2[1])
'''


































































































































































































# Lambda

def fn(a,b):
    return(a+b)

c = fn(10,20)
print(c)


c = fn(20,20)
print(c)




lfn = lambda a,b : a+b

c = lfn(10,20)

c





































































































































































