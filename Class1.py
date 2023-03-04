# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 21:40:22 2023

@author: vikas
"""

a = 10
#F9
b = 20
c = a + b
print(c)


print?
help(print)


import pandas

pandas.__version__


import pandas as pd

pd.__version__


#print

print("Hello")

print("Hello World")

print("Hello World $ !")


print()

print("Hello","World","Python")

help(print)

print("Hello","World","Python", sep="-!")


print("Hello")
print("World")
print("Python")

print("Hello", end='--')
print("World", end='++')
print("Python")

print("Hello","World","Python", sep="-!", end='**')
print("Hello", end='--')
print("World", end='++')
print("Python")



# Comments
# Single line using #
# multiple line using
'''
Variables in Python
1. String
2. Number - integer and float
3. boolean - True or False
'''

a = 10
b = 20
c = a+b

print("value of a is ", "a")

print("value of a is",a, "value of b is", b, "value of c is", c)

print("value of a is",a, "value of b is", b, "value of c is", c, sep='-' )

print("value of a is {a} value of b is {b} value of c is {b}")

print(f"value of a is {a} value of b is {b} value of c is {b}")

print("value of a is {0} value of b is {1} value of c is {2}".format(a,b,c), sep='--')



a = 10
print(type(a))

a = "10"
print(type(a))

a = 10.4
print(type(a))


#Type Conversion or Type Casting

i = 10

#Concatination
s = "Marks are " + i
s = "Marks are " + str(i)
print(s)


i = 10
print(str(i))
print(float(i))

f=10.3
print(str(f))
print(int(f))

s = "10.2"
print(int(float(s)))
print(float(s))


c = 'a'
ord(c)

c = '='
ord(c)

c = '!'
ord(c)


pas = "E"
if (ord(pas)>=97 and ord(pas)<=122):
    print("Pass Correct")
else:
    print("Pass Not Correct")

a = 97
chr(a)

a = 100
chr(a)

a = 125
chr(a)





























































