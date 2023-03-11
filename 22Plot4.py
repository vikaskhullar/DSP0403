#Line plot
# -*- coding: utf-8 -*-
# Interfaces in the plots
#-----------------------------
#%
#library
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 100)
x


fig = plt.figure()  #plot Figure
fig

#MATLAB interface
#create 1st panel
#(row, column, panel)
plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x))
#create 2nd panel
#(row, column, panel)
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))



plt.subplot(1, 2, 1)
plt.plot(x, np.sin(x))
#create 2nd panel
#(row, column, panel)
plt.subplot(1, 2, 2)
plt.plot(x, np.cos(x))

#%%%
#how to add something into the first one, overlap


#first create grid of plots
fig1, ax1 = plt.subplots(2)
#call plot() method on the appropriate object
ax1[0].plot(x, np.sin(x))
ax1[1].plot(x, np.cos(x))


print (fig1)
# switching between
plt.plot() and ax.plot() 
#for best visualisation


# -*- coding: utf-8 -*-
#
#-----------------------------
#%
# -*- coding: utf-8 -*-

#Python has excellent libraries for data visualization. A combination of Pandas, numpy and matplotlib can help in creating in nearly all types of visualizations charts. In this chapter we will get started with looking at some simple chart and the various properties of the chart.
##Creating a Chart
#We use numpy library to create the required numbers to be mapped for creating the chart and the pyplot method in matplotlib to draws the actual chart.
#import libraries
import numpy as np 
import matplotlib.pyplot as plt 
#create sample data
x = np.arange(0,10) 
x
y = x ^ 2 
y
#Simple Plot
plt.plot(x,y)
#line plot by default for numerical values

#Labling the Axes: we can apply labels to the axes as well as a title for the chart using appropriate methods from the library as shown below.

import numpy as np 
import matplotlib.pyplot as plt 
#%%%
x = np.arange(0,10) 
y = x ^ 2 
x,y
#Labeling the Axes and Title
#run these 4 lines together
plt.title("Graph Drawing") 
plt.xlabel("Time") 
plt.ylabel("Distance") 
plt.plot(x,y)
#see label values: title, x&y label


#Matplot Lib -Features
#-----------------------------
#%

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,1000)
x
#just the plot
plt.plot(x, np.sin(x))

plt.plot(x, np.tan(x))

#Axis - manual
plt.plot(x, np.sin(x))
plt.xlim(-1, 15)
plt.ylim(-1.5, 1.5)

#Axis - Reverse
plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(1.5, -1.5)

#%%%
#Axis - Together -spelling AXIS
plt.plot(x, np.sin(x))
plt.axis([-1, 11,-2, 2])
#[xmin xmax ymin ymax]
#%%%
#Axis - tight
plt.plot(x, np.sin(x))
plt.axis('tight')

#Axis - equal aspect ratio
plt.plot(x, np.sin(x))
plt.axis('equal')


#%%%Label Plots
#titles & axis
plt.plot(x, np.sin(x))
plt.title('Sine Curve')
plt.xlabel('X Label')
plt.ylabel('Y Label')



0 9 A B C D E F

'''
'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'
'''

#Multiple Curves, single line label
plt.plot(x, np.sin(x), label='X Curve', color='#FFAF00')
plt.plot(x, np.cos(x), label='Y Curve', color='b')
plt.title('Sine and Cos Curve')
plt.ylabel("dhiraj Upadhyaya")
plt.xlabel("Anil Jadon")
plt.axis('tight')
plt.legend()



#axis plot
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,10,1000)
ax= plt.axes()
ax.plot(x, np.sin(x))
ax.set( xlim=(0,10) , ylim = (-2, 2), xlabel = 'X Label', ylabel = 'Y Label', title = ' Plot with Sin and Cos Curve');
