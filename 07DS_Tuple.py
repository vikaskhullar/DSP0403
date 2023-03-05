# -*- coding: utf-8 -*-
#Tuples - collection, ordered, unchangeable/ unmutatble, like list, round bracket
#indexing is supportable


#List           Mutable, Indexed, Unorderd, collection, square bracket
#Dictionary     Mutable, key-value paired, Unorderd, collection, curly bracket
#Tuples -       unmutatble, collection, ordered, round bracket


tuple1 = (2,1,3,7,5)
tuple1

#everying like list except changes
tuple1[0] = 99
tuple1[0]

#access
tuple1[2]

for i in tuple1: 
    print(i)

a=30
b=20

if (a>b) :
    print("A is greater")


tuple1 = ("Vikas", "Raman", "Amit")

'Khullar' in tuple1

if ('Raman' in tuple1) : 
    print('is present in tuple')



if 'amit' in tuple1 : 
    print('Amit is present in tuple')


abc={"1":1,"2":2,"3":3}
list(abc.values())

tuple1.append('Khullar')  #error, cannot add

tuple1
len(tuple1)

#methods in tuple
#count, index, ...

tuple1
tuple1.remove()  #cannot remove also


#single value
tuple2a = ('a','s')

type(tuple2a)  #error incorrect way


tuple2b = 'a',
type(tuple2)


#end of tuple
#where do we used it - list type of object where no changes are required
#list of gender, courses, categories, countries, list of values
#https://learnbatta.com/blog/python-working-with-tuple-data-type-41/
