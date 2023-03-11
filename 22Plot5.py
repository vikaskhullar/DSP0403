# -*- coding: utf-8 -*-
#Matplot - Scatter Plot
#-----------------------------
#%


import matplotlib.pyplot as plt

x= [1,3,4,6,9,10,12,5,2,3]

y= [3,9,6,5,8,8,2,3,1,8]


plt.scatter(x,y, color ='r', marker='*')
plt.scatter(y,x, color ='b', marker='o')


x,y
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 90, 100)
y = np.sin(x)
x
y

#plot

plt.scatter(x, y)

plt.scatter(x, y, color='r')

plt.scatter(x, y, c='r' , marker='*')



size = (30 * np.random.rand(100))

size



x= [1,3,4,6,9,10,12,5,2,3]

y= [3,9,6,5,8,8,2,3,1,8]

z =[40,50,30,50,60,70,120,10,50, 20] 

plt.scatter(x, y, c='r' , marker='o', s=z)


x = [0,2,4,6,8,10]
y = [0]*len(x)

s = [20*4**n for n in range(len(x))]

plt.scatter(x,y,s=s)
plt.show()

#matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, *, plotnonfinite=False, data=None, **kwargs)

#https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.scatter.html



#eg2
import matplotlib.pyplot
import pylab

x = [1,2,3,4]
y = [3,4,8,6]



size = (30 * np.random.rand(4))**2
size

plt.scatter(x,y, s=size)
matplotlib.pyplot.show()





#Pie Chart
#-----------------------------
#%

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#FF9090'
plt.rcParams['axes.labelcolor']= '#909090'
plt.rcParams['xtick.color'] = '#909090'
plt.rcParams['ytick.color'] = '#909090'
plt.rcParams['font.size']=8



sizes=[120,30,10]

labels = ['BBA', 'MBA','PHD']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True)
plt.show()




import matplotlib.pyplot as plt

sizes=[4,5,6,3,2]

labels = ['A', 'B','C', 'D', 'E']

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False)
ax1.axis('equal')
plt.show()


#%%%

plt


import matplotlib.pyplot as plt
labels = ['Male',  'Female']
percentages = [60, 40]
explode=(0.15,0)
#

color_palette_list = ['#f600cc', '#ADD8E6', '#63D1F4', '#0EBFE9', '#C1F0F6', '#0099CC']

fig, ax = plt.subplots()
ax.pie(percentages, explode=explode, labels=labels, colors= color_palette_list, autopct='%1.1f%%',  shadow=True, startangle=90,  pctdistance=1.2, labeldistance=1.4)
ax.axis('equal')
ax.set_title("Distribution of Gender in Class", y=1)
ax.legend(frameon=False, bbox_to_anchor=(0.2,0.8))
plt.show()

