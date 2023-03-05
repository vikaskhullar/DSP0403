# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 19:32:19 2023

@author: vikas
"""

'''
In c/ c++


sequence of values

1,3,2,4,5,8,7

int a[10]; create a array
then able to add elelments in that array
for 

'''

'''
list
'''


l1 = []

l2 = [2,3,4,1,5,7]

l2.sort()


#Sequences in Python

'''

List
Tupple
Set
Dictionary

# Hetrogeneous or homogeneous , Mutable, Ordered, Indexed
'''



# List


l1 = []

type(l1)

l2 = [1,4,3,7,6,8]

a = 10
print(a)

print(l2)

# Indexing

l2[0]
l2[1]
l2[2]
l2[3]
l2[4]
l2[5]
l2[6]


#Hetrogeneous

l3 = [1, 3.4, True, 'Python']
print(l3)


emp = [901, 'VK', 'CSE', 'M', True]
emp

type(emp)

type(emp[0])

type(emp[1])

type(emp[2])

type(emp[4])



# individual access is possible through indexes

# List have 1000 entries and We want to access all entries individually

r4 = range(100)
r4

l4 = list(r4) #converting a range to list
l4


l4 = list(range(100))
l4


l5 = list(range(10, 21))
l5

l6 = list(range(10, 51, 5))
l6

l7 = list(range(10, 51, 5))

print("value i",l7[0])
print("value i",l7[1])
print("value i",l7[2])


# looping structre


'''
loop
{
startement 1
state 2
state 3 
}
'''


l7

for e in l7:
    print("value i",e)



for e in emp:
    print(e)


#Mutable allow to change any value

l2
l2[1] = 44
l2

l2.append(10)
l2

l2.remove(6)
l2

l2.pop()

l2

l2.pop()

l2.remove(44)
l2

l2.pop(1)


l2 = [2,5,7,5,4,3,8,9,7,5]

l2.count(5)
l2.count(3)
l2.sort()

l2


l2.reverse()
l2

l2.insert(4, 6)
l2





# Set
'''
unique data
ordered data
'''

s1 = {0}

s2 = {3,2,4,1,2,6,4,3}
s2 #Unique and Ordered


# Non-Indexed
s2 = {3,2,4,1,2,6,4,3}
s2

s2[0] #TypeError: 'set' object is not subscriptable Error

for s in s2:
    print(s)
    


# Mutable 

s2.add(5)    
s2
s2.pop()

s2.remove(5)
s2

s2.update({3,2,6,5,9,8,1})
s2


s2.remove(5)
s2
s2.remove(5) # KeyError: 5

s2
s2.discard(4)
s2
s2.discard(4)
s2




teamA = {'India', 'Australia','Pakistan', 'England'}
teamB = {'Bangladesh', 'New Zealand', 'West Indies', 'India'}
teamA
teamB

teamA.union(teamB)

teamA.intersection(teamB)



E = {1,2,3,4,5,11,12,13,14,15}
B = set(range(11,16))

E
B

A = E.difference(B)
A




A= set(range(1,11))
B =set(range(6,15))


A
B

E = A.union(B)

E


A
B
I = A.difference(B)
I


import numpy as np
TotalEmp = list(range(1,21))
TotalEmp
BenchEmp = np.random.choice(TotalEmp, size=10)
TotalEmp = set(TotalEmp)


TotalEmp
BenchEmp

WorkEmp = TotalEmp.difference(BenchEmp)
WorkEmp


# Dictionary

d1 = {}

# Key : Value Paired
d2 = {'empcode':1, 'empname': 'VK', 'empstatus': True}
d2

#Not Indexed

d2['empcode']

d2['empname']

d2['empstatus']

d2['empsal'] #KeyError: 'empsal'

d2.keys()

d2.values()

d2.items()

#Not Ordered

#Hetrogeneous
d2 = {'empcode':1, 2: 'VK', 'empstatus': True}
d2



car = { 'brand':'Honda', 'model': 'Jazz', 'year' : 2017}



d2.keys()
list(d2.keys())
set(d2.keys())

d2.values()

d2.items()



a = (1,2) # Tuple
b =(2,3)
c= (5,5)

l1 = [a,b,c]
l1

s1 = {a,b,c}
s1

a = [1,2] # Tuple
b =[2,3]
c= [5,5]

l1 = [a,b,c]
l1


car = { 'brand':'Honda', 'model': 'Jazz', 'year' : 2017}

car

car['brand']


#Mutable

car['year'] = 2020
car

car['sold'] = 2021

car


car.pop('year')
car

car.popitem()

car

car.values()



import numpy as np
rno = list(range(1,11))
name = ['Student'+str(i) for i in range(1,11)]
course = list(np.random.choice(['BBA', 'MBA', 'BTech', 'MTech'], size=10))


rno
name
course


student = {'RollNo': rno, 'Name': name, 'Course': course}

student

student.keys()

student['Name']

student['Name']

type(student['Name'])

student['Name'][0]
student['Name'][9]

student

import pandas as pd

df = pd.DataFrame(student)
df.to_csv('Student.csv')



# Tuple

t1 = ()

t2 = (2,3,5,1,7,6,7,8,3)
t2

#Not Ordered, Not Unique Values

#Indexed

t2[0]
t2[7]


#Not Mutable

t2[0] = 20 #TypeError: 'tuple' object does not support item assignment





# Conditional Statements

#if statement



a = 10
b = 20
a<b

if a<b:
    print("a is lesser")


a = 20
b = 10
a<b

if a<b:
    print("a is lesser")



# if else statement

a = 30
b = 20

if a<b:
    print("A is lesser")
else:
    print('b is lesser')


# if elif else

'''
Marks Grading System

M > 90              - A
M > 80 and M <= 90  - B
M > 70 and M <= 80  - C
M > 60 and M <= 70  - D
M > 50 and M <= 60  - E
M <= 50             - F
'''


M = 49

if M>90:
    print("A")
elif M > 80 and M <= 90:
    print("B")
elif M > 70 and M <= 80:
    print("C")
elif M > 60 and M <= 70:
    print("D")
elif M > 50 and M <= 60:
    print("E")
else:
    print("Fail")



M=65

if M>90:
    print("A")
    print("very good grades")
if M > 80 and M <= 90:
    print("B")
if M > 70 and M <= 80:
    print("C")
if M > 60 and M <= 70:
    print("D")
if M > 50 and M <= 60:
    print("E")
if M <=50:
    print("Fail")



#Looping Stuctures or Iterative Statements

l1 = list(range(0,21, 5))

l1

#for loop

for i in l1:
    print(i)




for i in l1:
    print("Value is",i)
    print("Multple of 2 is",i*2)
    print("Power of 2 is",i**2)
print('out')


# Write a table of 2 using for loop
'''
2 x 1 = 2
2 x 2 =4
-
-
-
2 x 10 = 20
'''


for i in range(1,11):
    print("2 x",i,"=",2*i)


for i in range(1,11):
    print(f"2 x {i} = {2*i}")


for i in range(1,11):
    print("2 x {0} = {1}".format(i,2*i))


# Munlti Loops

# Print  table of 2 to 4


for j in range(2,5):
    for i in range(1,11):
        print(f"{j} x {i} = {j*i}")



# While Loop

'''
While will work till condition is Boolean True
While is condition controlled loop
'''

#Print 1-10

cnt=1

while (cnt<=10):
    print(cnt)
    cnt = cnt + 1



cnt=1

while (True):
    print(cnt)
    cnt = cnt + 1




# Write a table of 2 using while loop
'''
2 x 1 = 2
2 x 2 =4
-
-
-
2 x 10 = 20
'''


i = 1
while (i<=10):
    print("2 x",i,"=",2*i)
    i=i+1



j=2
while(j<=4):
    i = 1
    while (i<=10):
        print(f"{j} x {i} = {j*i}")
        i=i+1
    j=j+1



# break and continue


lst = [3,2,6,5,8,9,1,7]

chk=5

for i in lst:
    print(i)
    if(chk==i):
        print(i,"element Found")



lst = list(range(1,10))
chk=5

for i in lst:
    if(chk==i):
        print(i,"element Found")
    print(i)
print("Completed")



lst = list(range(1,10))
chk=5

for i in lst:
    if(chk==i):
        print(i,"element Found")
        break
    print(i)
print("Completed")


# Continue

lst = list(range(1,10))
chk=5

for i in lst:
    if(chk==i):
        print(i,"element Found")
        continue
    print(i)
print("Completed")






#password entry system that allows three attempts



pswd = 99

cnt=1

while(True):
    en = int(input("Enter Password"))
    if pswd == en:
        print("Paswword Correct")
        break
    cnt = cnt + 1
    print("Password Incorrect")
    if cnt<=3:
        print(cnt, " - Re Chance to Enter Password")
        continue
    print("Exited")
    break




# Functions

'''
System Defined
User Defined

Reduce the line of written code
Enhance reutilization of written code
'''


def f1():
    print("Hello")



f1()

f1()

f1()



def compute():
    x = 10
    y = 20
    print(x+y)
    print(x-y)
    print(x*y)
    print(x//y)


compute()

compute()


#Function take inputs in the form of Parameters



def compute1(x,y):
    print(x+y)
    print(x-y)
    print(x*y)
    print(x//y)



compute1(10,20)

compute1(30,20)

compute1(40,50)




def empdet(eno, ename, dept, email):
    print(eno, ename, dept, email)



empdet(111, 'Vikas', 'CSE', 'v.k@g.com')


empdet(111, 'Vikas', 'CSE')




help(print)


#Default Values as paramerters or arguments

def empdet(eno, ename, dept='NA', email='NotAvailable'):
    print(eno, ename, dept, email)



empdet(111, 'Vikas', 'CSE', 'v.k@g.com')

empdet(111, 'Vikas', 'CSE')




#Return or Ouput from a function



def maximum(lst):
    m = 0
    for i in lst:
        if i>m:
            m=i
    print(m, "is greatest")


l1 = [4,3,6,9,2,1]
print(maximum(l1))

x = maximum(l1)

print(x)

print(maximum(l1))






def maximum(lst):
    m = 0
    for i in lst:
        if i>m:
            m=i
    print(m, "is greatest")
    return(m)
    


l1 = [4,3,6,9,2,1]
print(maximum(l1))

x = maximum(l1)
print(x)

























































































































































































































































































































































































































































































































































































