#Data Structure - List

# -*- coding: utf-8 -*-
#Data Structure - List

#List, Tuple, Set, Dictionary, Frozenset

# Hetrogeneous, Mutable, Ordered, Indexed



L1 = [1,2.6, 'Vikas', 'Khullar', True]
#mixed, ordered

L1
print(L1)

#index starts from 0 ; R starts from 1
L1[3]

#change values # mutable
L1[2] = 'YES'
L1

#ordered
L1[0], L1[1], L1[3], L1[4]
L1[5]  #does not exist
#another way to print
L1 = [1,2, 'Vikas', 'Khullar', True]


for i in L1:
    print(i)

#see indentation in line above
for i in L1: print(i) #same as obove

#range
range(5); range(len(L1))
for i in range(5) : print(i, end =' ')
#ends - how wrap the values
for i in range(len(L1)) : print(L1[i])

#other functions
L1 = [1,2, 3, 'Vikas', True, 'Vikas', "Vikas"]
L1[3]

L1[3]=L1[3].upper()

L1

L1[0:2]  #starts at 0, ends at 1 position



sum(L1[0:4])

#Hold
#L1.count(L1[4])


L1.count('Vikas')


L1.count('Khullar')  #how many times Khullar found

len(L1)  #no of elements


L1.append('Neha')
L1

L1.remove(2)
L1

L1.pop()

L1.pop()  #remove last index values

L1.pop(0)  #remove 0th position

L1


del L1[0]  #removes index value
L1


L1.clear()  #clear all values
L1


# Copy List
L1 = [1,2, 'Vikas', 'Khullar', True]
L1
#method 1
L2 = L1
L2

L1.append('Aman')
L2, L1
#The two variables are referencing to same location
L1.append('Vikas')
L2, L1


#method2
L1 = [1,2, 'Vikas', 'Khullar', True]
L3 = L1.copy()
L1.append('Aman')
L3, L1

L1.append('Aman')
L1
L3  #Aman not here

#method3

L1 = [1,2, 'Vikas', 'Khullar', True]
L4 = list(L1)
L1, L4

L1.append('Aman')
L1, L4

#Methods in list
#append, clear, copy, count, extend, index, insert, pop, remove, reverse, sort

L1= list(range(11))
L1


L6=[]
for i in L1:
    L6.append(i*i)

L6

#another list
L5 = [i*i for i in range(11)]
L5


L5=L1
L5

L5.reverse()
L5


L5=[3,5,2,6,1]
L5

L5.sort()
L5
#inplace sorting, permanent changes

#Ex

ruits.sort()
fruits
#put mango in 2nd position
fruits.insert(1, 'mango')
fruits
fruits.sort()
fruits


first=50
end=100

range(100)

range(50, 100)



#Join two lists
L1
L5
L6 = L1 + L5
L6

#https://learnbatta.com/blog/python-working-with-lists-51/
