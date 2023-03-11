# -*- coding: utf-8 -*-
#Functions
#Block of code which only runs when it is called
#can pass data, known parameters into function
#it can return data as result
#defined using keyword def with colon :



#declaration or defination of function

def oper():
    print(a+b)
    print(a-b)
    print(a*b)



#calling that declared Functions


a=10
b=20



oper()

oper()

oper()


def printHello():
    print("Hello ")
    
    
printHello()


def printHello2(fname, lname):
    print("Hello \t" + fname + lname)


printHello2() #error


printHello2('vikas', 'khullar')



def printHello3(name='Student'):
    print("Hello \t" + name)


printHello3()  #without value

printHello3('Dhiraj')



def printdef1(a,b):
    print("name "+a+b)

printdef1("Amit", "Saho")



L1=['a','b','c']

L2=['d','e','f']

L3=['g','h','i']


for i in L1:
    print("Hello \t" + i)



for i in L2:
    print("Hello \t" + i)
    
    

for i in L3:
    print("Hello \t" + i)
    
    
#passing a list 
def printHello4(names):
    for i in names:
        print("Hello \t" + i)

printHello4(L1)

printHello4(L2)

printHello4(L3)



printHello4()
printHello4('Dhiraj')

printHello4(['Dhiraj','Kounal','Pooja'])

namelist  = ['Student1','Student2','Student3']
printHello4(namelist)


#retun values

def totalSale(x=0,y=0):
    return(x + y)
    
totalSale()
totalSale(5,10)



"""Define a function reverse() that computes the reversal 
of a string. For example, reverse("I am testing") should 
return the string "gnitset ma I"."""

def reverse(str):
  return str[::-1]

#test
print (reverse("I am testing"))
print reverse("It's cool, isn't it?")



"""The function max() from exercise 1) and the function 
max_of_three() from exercise 2) will only work for two 
and three numbers, respectively. But suppose we have a 
much larger number of numbers, or suppose we cannot tell 
in advance how many they are? Write a function max_in_list() 
that takes a list of numbers and returns the largest one."""

def max_in_list(lst):
  max = 0
  for n in lst:
    if n > max:
      max = n
  return max

#test
print (max_in_list([2,3,4,5,6,7,8,8,9,10,1,5]))

print max_in_list([290,2,3,4])
print max_in_list([9,2,3,4,19])

