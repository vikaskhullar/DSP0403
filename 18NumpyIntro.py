# -*- coding: utf-8 -*-
#Numpy - Accessing Elements
#-----------------------------
#%


import numpy

import numpy as np



np.random.randint(100,1000)


#scalar
x0 = np.random.randint(10);
x0

#1-Dim
x1 = np.random.randint(100,200, size=10)
x1

#2-Dim
x2 = np.random.randint(10, 20, size=(3,4))
x2

sh=x2.shape

sh[0]
sh[1]

#3-Dim
x3 = np.random.randint(10, 20, size=(6,4,5))
x3


#Print those values
x0; x1;x2;x3


x0

#0-Dim

x0[0]

x0[1]  #error

x0  #this will work


x1

x1[2]

x1[1:4]

x1[3:]

x1[:4]

x1[:-2]

x1[-5:-2]


x2 = np.random.randint(10, 20, size=(5,5))
x2                       

x2[2][2]
x2[4][3]

x2

#2-Dim
x2[1]

x2[0,0]
x2[0,1]  #1st row, 2nd col
x2[0,2]
x2[0,3]

x2[1,0]
x2[1,1]  #1st row, 2nd col
x2[1,2]
x2[1,3]

x2

x2[0:,0:]  #all

x2  #full array


x2[0:2,0:2]  #1st row

x2
x2[4,3]


x2[:1, 2:4]

x2[:,1:2]  #1st col

x2[:, :1]


x2[:2, :3] #first 2 rows, first 3 columns

x2

x2[1,-2:]

x2

x2[1,-2]
x2



'''
#alternate items
x2[::, ::] #all

x2[::2, ::] #alternative rows

x2[::, ::2] #alternative cols

x2
#3-Dim
x3  #all
x3[::] #all
x3[1::] #2nd matrix onwards
x3[::2]  #alternative matrix
x3[1:2:] #2nd matrix
x3[0,0,0]  #first cell
x3[2,3,4]  #last cell of array
x3[0]  #first matrix
x3[0,1]  #first matrix, 2nd row
x3[1,2] #2nd matrix, 3rd row
x3
'''


#Topic: Numpy  - array
#-----------------------------
#libraries

#the heart of a Numpy library is the array object or the ndarray object (n-dimensional array). You will use Numpy arrays to perform logical, statistical, and Fourier transforms. As part of working with Numpy, one of the first things you will do is create Numpy arrays. 
#There are three different ways to create Numpy arrays:
#Using Numpy functions
#Conversion from other Python structures like lists
#Using special library functions

import numpy as np

#Creating a One-dimensional Array
import numpy as np

array = np.arange(20)
array

array1 = np.arange(5,20)
array1

array2 = np.arange(0,100, step=5)
array2

sh =array2.shape
sh[1]


array[3]

#Numpy Arrays are mutable, which means that you can change the value of an element in the array after an array has been initialized.
array[3] = 100
print(array)


array2 = np.arange(0,20)
array2.shape

#Creating a Two-dimensional Array
#If you only use the arange function, it will output a one-dimensional array. To make it a two-dimensional array, chain its output with the reshape function
array2reshaped = array2.reshape(4,5)

array2
array2reshaped

array2.shape
array2reshaped.shape

asingledim = array2reshaped.reshape(20,)

#Since we get two values, this is a two-dimensional array. To access an element in a two-dimensional array, you need to specify an index for both the row and the column.
array2[3][4]

#Creating a Three-dimensional Array and Beyond
array3 = np.arange(27).reshape(3,3,3)
array3
#The number of elements in the array (27) must be the product of its dimensions (3*3*3). To cross-check if it is a three-dimensional array, you can use the shape property.
array3.shape


#arange
#using the arange function, you can create an array with a particular sequence between a defined start and end values
np.arange(10, 35, 3)


#%%
#Using Other Numpy Functions
#Other than arange function, you can also use other helpful functions like zerosand ones to quickly create and populate an array.
#Use the zeros function to create an array filled with zeros. The parameters to the function represent the number of rows and columns (or its dimensions).


a=0
b=1

a=a+b
a

import numpy as np

a=np.zeros((2,4))
a

b=np.ones((2,4))
b

a=a+b
a


#The empty function creates an array. Its initial content is random and depends on the state of the memory.

np.empty((2,3))


# eye function lets you create a n * n matrix with the diagonal 1s and the others 0.
np.eye(3,3)

#function linspace returns evenly spaced numbers over a specified interval. For example, the below function returns four equally spaced numbers between the interval 0 and 10.

np.linspace(0, 10, num=4)

#%%
#Conversion from Python Lists
#Other than using Numpy functions, you can also create an array directly from a Python list. Pass a Python list to the array function to create a Numpy array:
array5 = np.array([4,5,6])
array5
#create a Python list and pass its variable name to create a Numpy array.
list1 = [4,5,6]
list1
array6 = np.array(list1)
array6
#both the variables, array and list, are a of type Python list and Numpy array respectively.
type(list1)
type(array6)

#%%%
#To create a two-dimensional array, pass a sequence of lists to the array function.
array7 = np.array([(1,2,3), (4,5,6)])
array7
array7.shape

#%%Using Special Library Functions
#You can also use special library functions to create arrays. For example, to create an array filled with random values between 0 and 1, use random function. This is particularly useful for problems where you need a random state to get started.
np.random.random((2,2))

#Creating and populating a Numpy array is the first step to using Numpy to perform fast numeric array computations. Armed with different tools for creating arrays, you are now well set to perform basic array operations.

#Links
#https://www.pluralsight.com/guides/different-ways-create-numpy-arrays
#