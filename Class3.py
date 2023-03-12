# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 19:22:28 2023

@author: vikas
"""

'''Rules for Python variables:

A variable name must start with a letter or the underscore character
A variable name cannot start with a number
A variable name can only contain alpha-numeric characters and underscores (A-z, 0-9, and _ )
Variable names are case-sensitive (age, Age and AGE are three different variables)
'''

import numpy as np

np.random.randint(100)

x1 = np.random.randint(100, 1000)
x1

x2 = np.random.randint(10, 100, size=10)
x2

x3 = np.random.randint(10,100, size=(5,4))
x3


#Numoy is Sequence type of Data Type

#Numpy Properties
#Homogeneous, Either integer, float, boolean
#Indexed

x2[0]


 
x4 = np.random.randint(10,20, size=(3,4,5))
x4


x5 = np.random.randint(10,20, size=(10,10,10,10))
x5



#Indexing

x2 = np.random.randint(10, 100, size=10)
x2

x2.shape

x2[0]

x2[0:4]

x2[0:6]

x2[:6]

x2[2:6]

x2[2:]

x2[-1]

x2[:-2]
x2

x2[4:-3]


x3 = np.random.randint(10, 100, size=(5,7))
x3

x3[0]

x3[0][0]

x3[0]
x3[0][1]

x3
x3[1]
x3[1][0]

x3[0][-1]

x3[0][-3:]

x3[-1]

x3[-1,-3:]


x3

x3[:,0]

x3[:,:2]

x3[:,-1]

x3[:, 2:4]
x3
x3[2:4, 2:4]



x4 = np.random.randint(0, 100, size=(3,4,5))
x4


x4[1,1:,-2:]

x3

x3[::2]

x3[::3]

x3[:,::2]


x3[::2]

x3

x3[:,::2]



#Functions

r1 = np.arange(10)

r2 = np.arange(10,20)
r2

r3 = np.arange(10, 100, 5)
r3


x3.shape

x4.shape



x3

x3r = x3.reshape((7,5))

x3r
x3.shape


x3r = x3.reshape((6,3))
x3r


x4 = np.zeros((3,4))
x4

x5 = np.random.randint(3,10, (3,4))

x5

x5 + x4


x6 =  np.ones((3,4))
x6

#By default Data type of numpy element is float


np.eye(3,3)


np.eye(5,3)

np.linspace(0,10, num=4)

x5 = np.random.randint(10,20, size=(3,3))
x5

xi = np.eye(3,3)

x5*xi





l1 = [1.1, 4.3, 6.7]
l1

np.array(l1, dtype='float')



x6 = np.random.randint(1, 100, size=10000)

x6

x6.shape


np.mean(x6)
np.median(x6)
np.std(x6)
np.var(x6)
np.max(x6)
np.min(x6)

x6 = np.random.randint(1, 100, size=(4,5))

x6
np.mean(x6)
np.median(x6)
np.std(x6)
np.var(x6)
np.max(x6)
np.min(x6)


x6
np.mean(x6, axis=0)
np.mean(x6, axis=1)

np.median(x6, axis=0)
np.median(x6, axis=1)

np.std(x6, axis=0)
np.std(x6, axis=1)

np.var(x6)
np.max(x6)
np.min(x6)



x6 = np.random.randint(1, 20, size=(2,3,4))
x6
np.max(x6)
np.max(x6, axis=0)
np.max(x6, axis=1)
np.max(x6, axis=2)


np.floor([1.2, 1.6])

np.ceil([1.2, 1.6])

np.round([1.2, 1.6])

np.trunc([1.2, 1.6])


np.floor([-1.2, -1.6])

np.ceil([-1.2, -1.6])

np.round([-1.2, -1.6])

np.trunc([-1.2, -1.6])


np.round([1.23456, 1.4325667], 2)

np.round([1.23456, 1.4325667], 0)




x1 = np.random.randint(1, 10, size=(3,4))

x2 = np.random.randint(11, 20, size=(3,4))

x1
x2

x3 = np.concatenate([x1,x2], axis=0)
x3


x1
x2
x4 = np.concatenate([x1,x2], axis=1)
x4


x3

x5 = np.split(x3,2, axis=1)
x5

x6 = np.split(x3,2, axis=0)
x6



x2 = np.random.randint(11, 20, size=(3,4))

x2
x2>14

x2[x2>14]



x2==14
x2[x2==14]




x2 = np.random.randint(0, 5, size=(3,4))
x2

x2!=0
x2[x2!=0]


x2 = np.random.randint(0, 15, size=(5,4))
x2

(x2>=8) & (x2<=11)

x2[(x2>=8) & (x2<=11)]

xn = (x2>=8) & (x2<=11)
xn
x2
xs = x2 * xn
xs

x2
x3 = x2.T
x3



#Pandas


import pandas as pd
xl = pd.ExcelFile('20denco.xlsx')
for sn in  xl.sheet_names:
    xls = pd.read_excel('20denco.xlsx', sheet_name=sn)
    print(xls*10)
    
xls = pd.read_excel('20denco.xlsx')
xls    






import pandas as pd
import numpy as np

for i in range(1,5):
    marks = np.random.randint(0+i,101+i, size=10)
    rno=np.arange(1+i,11+i)
    df = pd.DataFrame({'rollno':rno, 'marks':marks})
    df.to_csv('test/'+str(i)+'.csv')



import glob
file = glob.glob('test/*.csv')
df = pd.read_csv(file[0])
for f in file[1:]:
    df = pd.concat([df,pd.read_csv(f)])
df.to_csv('test//final.csv')



import pandas as ps

from pydataset import data
data()
mt = data('mtcars')
mt


type(mt)
mt.to_csv('mt.csv')


import pandas as pd

df = pd.read_csv('mt.csv')

df

df = data('mtcars')

df.columns
df

df.index

df['mpg']

df.loc['Mazda RX4']

df

df.head()
df.tail()




import pandas as pd
s1 = pd.Series([1,3,4,2,5])
s1

import numpy as np
n1 = np.random.randint(1,100, size=5)
n1
s2 = pd.Series(n1)
s2

s2[0]
s2[0:2]

# System Defined Index Must Have Properties, 1. 
#start with 0, sequence range 0,1,2,3..., must be uniqe


s3 = pd.Series(n1, index=['a','b','e','c','a'])
s3

s3['a']
s3['b']
s3['a']

s3.loc['a']


s3.iloc[0]
s3.iloc[1]
s3.iloc[2]

s3

s3.values
s3.index

s3.index = range(101, 106)
s3


n1 = np.random.randint(1,100, size=10)
n1
s4 = pd.Series(n1)
s4

s4>50

s4[s4>50]




import pandas as pd
course = pd.Series(['BTech', 'MTech', 'MBA', 'BBA'])
strength = pd.Series([100, 50, 150, 200])
fee = pd.Series([2.5, 1.5, 2, 2.7])

course
strength
fee

d1 = {'Course':course, 'Strength':strength, 'Fee':fee}
df = pd.DataFrame(d1)
df
df.to_csv('stud1.csv')

df.index

df.index = df['Course']

df

df = df.drop(['Course'], axis=1)

df

df.columns
df.index



df['Strength']

df.index=='BBA'

df[df.index=='BBA']


df

df['Strength']>100

df[df['Strength']>100]


df
df[(df['Strength']>100) & (df['Fee']==2.7)]
df[(df['Strength']>100) | (df['Fee']==2.7)]



#Data Cleaning

pd4 = pd.DataFrame([['dhiraj', 50, 'M', 10000, None], ['Vikas', None, None, None, None], ['kanika', 28, None, 5000, None], ['tanvi', 20, 'F', None, None], ['poonam',45,'F',None,None],['upen',None,'M',None, None]])
pd4

pd4.dropna()

pd4.dropna(axis=0)
pd4.dropna(axis=1)

pd4.dropna(axis='rows')
pd4.dropna(axis='columns')

pd4
pd4.dropna(axis=0, how='all' )
pd4.dropna(axis=1, how='all')


pd4
pd4.dropna(axis=0, how='any' )
pd4.dropna(axis=1, how='any')


pd4
pd4.dropna(axis=0, thresh=3 )
pd4.dropna(axis=1, thresh=3)



import pandas as pd
df = pd.read_csv('AirPassengers.csv')[:40]
df
df.plot()
df
df.dropna()

df.fillna(0)
df.fillna(0).plot()

df.fillna(method='ffill')
df.fillna(method='ffill').plot()


df = pd.read_csv('AirPassengers.csv')[:20]
df
df.fillna(method='ffill')




import pandas as pd

eno = pd.Series(range(1,11))

ename = ['Emp'+str(i) for i in range(1,11)]
ename


'''
ename=[]
for i in range(1,11):
    ename.append('Emp'+str(i))
ename
'''

genderlist = np.random.choice(['M','F'], size=10)
genderlist

dlist = np.random.choice(['HR', 'Mech', 'Civil', 'Manager'], size=10)
dlist

sal = np.random.choice([10,15, 20, 12], size=10)
df = pd.DataFrame({'Eno':eno, 'Ename':ename, 'gender':genderlist, 'Department':dlist, 'Salary':sal})
df




#10000 Entries
eno = pd.Series(range(1,10001))

ename = ['Emp'+str(i) for i in range(1,10001)]
ename


'''
ename=[]
for i in range(1,11):
    ename.append('Emp'+str(i))
ename
'''

genderlist = np.random.choice(['M','F'], size=10000)
genderlist

dlist = np.random.choice(['HR', 'Mech', 'Civil', 'Manager'], size=10000)
dlist

sal = np.random.choice([10,15, 20, 12], size=10000)
df = pd.DataFrame({'Eno':eno, 'Ename':ename, 'gender':genderlist, 'Department':dlist, 'Salary':sal})
df.to_csv("Emp.csv")































































































































































































































