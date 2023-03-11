#Bar Plot

# -*- coding: utf-8 -*-
#Created on Fri Jun 14 19:31:48 2019 @author: dhiraj@dell sip
#Graph and Plots
#-----------------------------
#%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.bar(['M','F'], [30,10])

plt.barh(['M','F'], [30,10])

plt.bar(['M','F'], [30,10], color = ['r', 'g'])

#with other options
plt.bar(['M','F'], [30,10], color = ['r', 'g'])
plt.title("Student Proportion")
plt.grid(True)
plt.show()  #used if running as file


#now create a dataframe object
df = pd.DataFrame({'gender':['MALE','F'], 'strength':[25,10]})
df

df.plot(kind='bar')  #legend, labels ?

plt.bar(df['gender'], df['strength'])

#save in object
df.columns

ax = df.plot(kind='bar', grid=True, legend=True, title=' Title of Graph', figsize=[8,3], color= ['red','blue'])
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
ax.set_facecolor('blue')


#M2
plt.bar(df.gender, df.strength) #use column anmes
plt.bar(df.gender, df.strength, color = ['red','g'])  #list color
plt.bar(df.gender, df.strength, color = ('red','g'))  #tuple color

ls = np.linspace(0,100, 5 )
ls

#complete plot
plt.bar(df.gender, df.strength, color = ['red','g'], width=0.2)
plt.title('Str of Students',fontsize=40)
plt.grid(True)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(df.gender,rotation=90,fontsize=20)
plt.yticks(rotation=90,fontsize=20)



ls = np.linspace(0,100, 6)
ls
#complete plot
plt.bar(['s1', 's2'], [80, 50], color = ['red','g'], width=0.2)
plt.title('Str of Students',fontsize=20)
plt.grid(True)
plt.xlabel('student name')
plt.ylabel('percentage')
plt.xticks(rotation=90,fontsize=10)
plt.yticks(ls, rotation=90,fontsize=10)


ls = np.linspace(0,100, 11)
ls
plt.show()

#what if both columns are numeric
# x axis values 
x = [1,2,3,4,5,6] 
# corresponding y axis values 
y = [2,4,1,5,2,6] 

plt.plot(x,y)
# plotting the points  
plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3,   marker='*', markerfacecolor='red', markersize=30) 
plt.plot(y, x, color='red',  linewidth = 7,   marker='o', markerfacecolor='red', markersize=12) 
plt.xticks(list(range(0,11)))
# setting x and y axis range 
plt.ylim(1,8) 
plt.xlim(0,5) 

# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
  
# giving a title to my graph 
plt.title('Some cool customizations!') 
  
# function to show the plot 
plt.show() 


#Subplots

#Bar Plot
#-----------------------------
#%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2)

fig

ax[0].barh(y=['CS','EC','ME'], width=[10,20,30])

ax[1].plot(x,y)

fig
