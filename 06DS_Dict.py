# -*- coding: utf-8 -*-
#Data Stuctures - Dictionaries
#Dictionary - collection, unordered, changeable/ mutable, indexed, curly bracket with key:value pairs

#List           Mutable, Indexed, Unorderd, collection, square bracket
#Dictionary     Mutable, key-value paired, Unorderd, collection, curly brackets
#Tuples -       unmutatble, collection, ordered, round bracket

stud={'rollno':1, 'name': "Vikas"}
stud['rollno']

stud={'name':'Vikas', 'class':'MBA'}

stud

car = { 'brand':'Honda', 'model': 'Jazz', 'year' : 2017}
car

#access
car['brand']

car.get('brand')

#change
car['year'] = 2018

car.get('year')
car


#list all Keys
for i in car: 
    print(i)
    

#list all values
for i in car: 
    print(car[i])

car.values()


for i in car.values() : 
    print(i)

car.items()

([('brand', 'Honda'), ('model', 'Jazz'), ('year', 2018)])
#key - value
for j, i in car.items() : 
    print(i, j)

Honda brand
Jazz model
2018 year


#check if key exist
if 'model' in car: 
    print('Model key exists')

#length
len(car)

#add items
car['color1']= 'black'
car


#remove items
pop_element =car.pop('color1')
pop_element
car

#removes last inserted item ie color here
car.popitem()
car


#remove item with specified key name
del car['brand']
car

#delete dictionary
del car
car

#empty dictionary
car = { 'brand':'Honda', 'model': 'Jazz', 'year' : 2017}
car

car.clear()
car

car['brand'] ='hyundia'
car

#copy dictionary
car = { 'brand':'Honda', 'model': 'Jazz', 'year' : 2017}
car

yourcar = car.copy()
yourcar

car['color']="Black"
yourcar
car


#method2
hiscar = car
hiscar

#new dictionary
newcar = dict(brand ='Honda', model ='Jazz', year=2017)
newcar
#round bracket, no quotes in keys, equal to symbol


#methods 
#clear, copy, fromkeys, get, items, pop, popitem, setdefault, update, values


car(['brand','year'])


student= {('name', 1):'Vikas',('name', 2):'Arun', ('Rank',2):'Pass'}

student[('name',2)]


student = {(1, 'Vikas'): 'Noida', (1,'Pooja'): 'Ghaziabad'}


type(student)

student[(1,'Vikas')]
student[(2,'Aman')] = 'Delhi'
student
student.keys()
student.values()
student.items()

#create a dictionary of phone : first,name -> mobile no

#uses : create dataframe, key-value pairs
