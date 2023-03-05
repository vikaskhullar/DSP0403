# -*- coding: utf-8 -*-
#if-else condition
#logical operations


if (condition)
   {
    
    
     }


if [conditional true then]:
    statement



= "assignment operator"

a=10
b=20
a=b
a
b


#conditional operators

== "check is equal to"

a=20
b=20

a==b

a<b

a>b

a<=b

a>=b

a!=b



x=2

if (x==2):
    
    print("right")


#short methods
if x == 2: print("x = 1")

#


if True then:
    state1
else
    state2

x=20
y=100

if (x>y):
    print("x is greater") # state 1
else:
    print("y is greater") #state 2
        



Marks=90



if(Marks>80):
    print("A")

elif (Marks>70 and Marks<=80):
    print("B")

elif(Marks>60 and Marks<=70):
    print("C")

else:
    print("F")





#longer way
if x > y:
    print("X")
elif x == y:
    print("=")
else :
    print("Y")



#other logical operations
#==, !=, <, > , <=, >= 

#with else 


if x == 1: 
    print(" x=1")
else:
    print("x not equal to 1")

x=2

if x == 1: 
    print(" x=1")
else:
    print("x not equal to 1")

#no bracket, use of indentation with colon operator
#elif

if x == 1: 
    print(" x=1")
elif x == 2:
    print(" x=2")
else :
    print("x not equal to 1 or 2")

x=3
if x == 1: 
    print(" x=1")
elif x == 2:
    print(" x=2")
else :
    print("x not equal to 1 or 2")


#shorthand if

#or and and


x=3; y=4; z=5

if x < y and y < z:
  print("Both conditions are True")


z=2

if x < y or y > z:
  print("Both conditions are True")



if (x < y) or (y > z) :
  print("Either conditions are True")



if ((x < y) or (y > z)) and (x > 10):
    print("Both conditions are True")
else:
    print("Conditions are not True")




if ((x < y) or (y > z)) or (x > 10):
    print("Either conditions are True")
else:
    print("Conditions are not True")

(x < y) or (y > z) and (x > 10)
x<y, x>z, x<10
True or False and True  #left to right

(x > 10) and (x < y) and (y > z)  

if (x < y) or (y > z) and (x > 10):
    print("Either conditions are True")
else:
    print("Conditions are not True")
