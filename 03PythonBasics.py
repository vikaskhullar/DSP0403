

#Topic : Basic Programming

#Numbers: Integers and floats work as you would expect from other languages:
x = 3

print(x) 

print(type(x))       # Prints "3"

print(x + 1)   # Addition; prints "4"
print(x - 1)   # Subtraction; prints "2"
print(x * 2)   # Multiplication; prints "6"
print(x ** 2)  # Exponentiation; prints "9"

x=5
x += 1
x
x=x+1

print(x)  # Prints "4"
x=5
x *= 2
print(x)  # Prints "8"

x=5
x **= 2
print(x)  # Prints "8"

y = 2.5
print(type(y)) # Prints "<class 'float'>"


print(y, y + 1, y * 2, y ** 2) # Prints "2.5 3.5 5.0 6.25"


#Python does not have unary increment (x++) or decrement (x--) operators
#%%
#Booleans: Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols (&&, ||, etc.):

t = True
f = False
print(type(f)) # Prints "<class 'bool'>"


AND
0 0 0
0 1 0
1 0 0
1 1 1

OR
0 0 0
0 1 1
1 0 1
1 1 1

Not
0 1
1 0


Ex OR
0 0 0
0 1 1
1 0 1
1 1 0

t=1
f=0

print(t and f) # Logical AND; prints "False"

print(t or f)  # Logical OR; prints "True"

print(not t)   # Logical NOT; prints "False"

t=1
f=1

print(t != f)  # Logical XOR; prints "True"

#%%

#Strings: Python has great support for strings:

Fname='vikas'
type(Fname)

Lname='khullar'

name = Fname +' ' + Lname

print(name)

h = 'hello'    # String literals can use single quotes
w = "world"    # or double quotes; it does not matter.

print(h)       # Prints "hell
print(len(h))  # String length; prints "5"

hw = h + ' ' + w  # String concatenation

print(hw)  # prints "hello world"


hw12 = '%s %s %s' % ('hello', 'world', 12)  
hw12


# sprintf style string formatting

print(hw12)  # prints "hello world 12"

#String objects have a bunch of useful methods; for example:

s = "hello"

s.capitalize()

print(s.capitalize())  # Capitalize a string; prints "Hello"

print(s.upper())       # Convert a string to uppercase; prints "HELLO"

print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"

print(s.center(10))     # Center a string, padding with spaces; prints " hello "

s=s.replace('e', 'yyy')
s

print(s)  # Replace all instances of one substring with another;   # prints "he(ell)(ell)o"

s=s.replace('o', 'yyy')
print(s)  # Replace all instances of one substring with another;   # prints "he(ell)(ell)o"


s = "hello"

z='  world '
print(z)
print(z.strip())  # Strip leading and trailing whitespace; prints "world"


#%%%Containers
#Python includes several built-in container types: lists, dictionaries, sets, and tuples.


#ListsA list is the Python equivalent of an array, but is resizeable and can contain elements of different types:

xs = [30, 10, "ff", 55, 888]    # Create a list
xs


print(xs[2])  # Prints "[3, 1, 2] 2"


print(xs[-2])     # Negative indices count from the end of the list; prints "2"

xs[2] = 'foo'     # Lists can contain elements of different types

print(xs)         # Prints "[3, 1, 'foo']"

xs.append('')


xs.append('bar')  # Add a new element to the end of the list

print(xs)         # Prints "[3, 1, 'foo', 'bar']"


x = xs.pop()      # Remove and return the last element of the list

x

xs
print(x, xs)      # Prints "bar [3, 1, 'foo']"

#Slicing: In addition to accessing list elements one at a time, Python provides concise syntax to access sublists; this is known as slicing:

x=range(5, 10)

x

a=100.6

b=a//10
b

import math
math.ceil(b)

a = range(10, 20)
type(a)

nums = list(a)     # range is a built-in function that creates a list of integers
nums

print(nums)               # Prints "[0, 1, 2, 3, 4]"

print(nums[2:5])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"

print(nums[5:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"

print(nums[:3])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"

print(nums)            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"

print(nums[-4:])          # Slice indices can be negative; prints "[0, 1, 2, 3]"

nums[2:4] = [8, 9]        # Assign a new sublist to a slice
nums
print(nums)               # Prints "[0, 1, 8, 9, 4]"



#Some Builtin Functions


a= bin(17)
a

a=bool(0)
a

a=bytearray(10)
a

#a=bytes(6)
a
ASCII

a=chr(65)
a



a=eval("False or False")
a

help()

a=hex(19)
a

x = iter(["apple", "banana", "cherry"])
x

print(next(x))
print(next(x))
print(next(x))

len(a)

max(iter(["apple", "banana", "cherry"]))

a=range(2,10)
a
list_a=list(a)
list_a

round(22.6)

a=str(11.7)
a

x=iter([1,4,2])

a=sum(x)
a

type(a)

abs(-11.7)

mylist = [True, True, False]
x = any(mylist)
x

len(x)

x = ['apple', 'banana', 'cherry']
len(x)



print('Enter your name:')
x = input()
x

x= input("Enter a number")
x

x = pow(4, 3)
x
