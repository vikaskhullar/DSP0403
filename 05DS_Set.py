# -*- coding: utf-8 -*-
#Set - collection, unordered, unindexed,curly bracket{},immutable, can add items


set= {1,2.5,'vikas', False, False, "Khullar", 50 }
set
studdetail = {111, "Vikas", 44.4, False}

print(set)

set1 = {1,2, 'SIP', 'Vikas', True}
set1

set1[0]  #cannot be accessed by index position, unordered

set1 ={1,2,3,4,5}

j=0

for a in set1:
   print(a)
    

for i in set1 : 
    print(i, end =' ')

#properties of set


'Dhiraj' in set
'Khullar' in set
2.5 in set

11 in set

#add item
set
set.add(11)
set

#sorted
#add duplicate name of Pooja
set.add('Khullar')
set

set.add(11)
set


set2.add('Pooja')
set2
#no duplicates

set_test={"vikas", "aman", "sam"}
set_test

set
set.update(['ABC', 22])
set

#adding more than 1 item
set.update(['ABC','DEF'])
set
set2

#set.update("XYZ") #this will add individual
set

len(set2)

set2 = {1,2,111,123,133, 'SIP', 'Dhiraj', False, 'SIP'}
set2
set2.remove('SIP') # cant utilize with a list
set2
set2.remove('Dhiraj')
set2


set1  #error if Pooja does not exist
set2.remove('Akhil')


set2.discard(1)  #no error now
set2

set1.discard('Kounal')  #remove if found
set1


set2
#pop
set2.pop()  #remove any location, unordered
set2
set2.pop()
set2

set2={1,2,3}
set2
set2.clear()  #empty
set2

del set2
set2

set1 = {1,2, 'SIP', 'Dhiraj', True}
set1


#other methods in sets
#add, clear, copy, difference, differenc_update(), discard, intersection, intersection_update, isdisjoint, issubset, issuperset, pop
#remove, symmetric_difference, symmetric_difference_update(), union, update

#Exercise

teamA = {'India', 'Australia','Pakistan', 'England'}
teamB = {'Bangladesh', 'New Zealand', 'West Indies', 'India'}
teamA
teamB
#add Sri Lanka to TeamA
#create a teamC with all teams
#Perform all the set operations 
teamA.union(teamB)
teamA.difference(teamB)
