#Topic: Line Plot
#-----------------------------
#libraries

import matplotlib
import matplotlib.pyplot as plt

# #structure
# plt.plot(xAxis,yAxis)
# plt.title('title name')
# plt.xlabel('xAxis name')
# plt.ylabel('yAxis name')
# plt.show();

#%% List Data
Year = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]
Unemployment_Rate = [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
Unemployment_Rate1 = [1.8,2,8,4.2,6.0,7,3.5,5.2,7.5,5.3]
Year
Unemployment_Rate



plt.plot(Year, Unemployment_Rate)
plt.title('Unemployment Rate Vs Year')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate')
plt.show()

#01FFAA

#styles
plt.plot(Year, Unemployment_Rate, color='#01FFAA', marker='o', label ="UR1")
plt.plot(Year, Unemployment_Rate1, color='#01FF00', marker='*', label = "UR2")
plt.title('Unemployment Rate Vs Year')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Unemployment Rate', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()





#%% Pandas DF
import pandas as pd
Data = {'Year': [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010], 'Unemployment_Rate': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]}
Data  
df = pd.DataFrame(Data,columns=['Year','Unemployment_Rate'])
df  

plt.plot(df['Year'], df['Unemployment_Rate'], color='red', marker='*')

plt.title('Unemployment Rate Vs Year', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Unemployment Rate', fontsize=14)
plt.grid(True)
plt.show()



#Topic: Bar Plot
#-----------------------------
#libraries

import matplotlib.pyplot as plt

#structure
# plt.bar(xAxis,yAxis)
# plt.title('title name')
# plt.xlabel('xAxis name')
# plt.ylabel('yAxis name')
# plt.show();

#%% List data
Country = ['USA','Canada','Germany','UK','France']
GDP_Per_Capita = [45000,42000,52000,49000,47000]

#run together
plt.bar(Country, GDP_Per_Capita)
plt.title('Country Vs GDP Per Capita')
plt.xlabel('Country')
plt.ylabel('GDP Per Capita')
plt.show();


#%%%
New_Colors = ['green','blue','purple','brown','teal']
plt.bar(Country, GDP_Per_Capita, color=New_Colors)
plt.title('Country Vs GDP Per Capita', fontsize=14)
plt.xlabel('Country', fontsize=14)
plt.ylabel('GDP Per Capita', fontsize=14)
plt.grid(True)
plt.show();

#%% DF data

import pandas as pd

Data = {'Country': ['USA','Canada','Germany','UK','France'], 'GDP_Per_Capita': [45000,42000,52000,49000,47000]    }
df = pd.DataFrame(Data,columns=['Country','GDP_Per_Capita'])
df

New_Colors = ['green','blue','purple','brown','teal']
plt.bar(df['Country'], df['GDP_Per_Capita'], color=New_Colors)
plt.title('Country Vs GDP Per Capita', fontsize=14)
plt.xlabel('Country', fontsize=14)
plt.ylabel('GDP Per Capita', fontsize=14)
plt.grid(True)
plt.show();





#Topic:
#-----------------------------
#libraries
import matplotlib.pyplot as plt

labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 35, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]
men_std = [2, 3, 4, 1, 2]
women_std = [3, 5, 2, 3, 3]

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

fig

ax.bar(labels, men_means, width, yerr=men_std, label='Men')
ax.bar(labels, women_means, width, yerr=women_std, label='Women')

ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.legend()

plt.show()


#Multiple Graphs
# First create some toy data:

import numpy as np
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# Create just a figure and only one subplot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)



#Topic:IIM  - Graph
#-----------------------------
#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt?

#%%basic scatter plot
for
plt.scatter(x, y)
plt.show();

#%%
#dataset
from pydataset import data
mtcars = data('mtcars')
#conda upgrade --all -y
df=mtcars

#%% scatter plot
#dim - x, y, shape(s), color(c), tranpsarency(alpha)
df.describe #summary
df.dtypes  #data types
df.columns
df['wt']; df['mpg']
plt.scatter(x='wt', y='mpg', data=df)
plt.show();

df.carb

df.carb.value_counts()
#color, transparency, shape
'''
x_axis_data- An array containing x-axis data.
y_axis_data- An array containing y-axis data.
s- marker size (can be scalar or array of size equal to size of x or y)
c- color of sequence of colors for markers.
cmap- cmap name
'''

plt.scatter(x='wt', y='mpg', s='hp', alpha=.5, c='carb', data=df)
plt.legend(loc='upper left')
plt.show();





#m = np.repeat([".", "<"], (df.carb.unique()).size)

#%%tips data
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() #default settings
tips_df = sns.load_dataset('tips')
tips_df
tips_df.columns
tips_df.total_bill
total_bill = tips_df.total_bill.to_numpy()
tip = tips_df.tip.to_numpy()
tip
plt.scatter(total_bill, tip)
plt.show();

#%% labels
plt.scatter(total_bill, tip)
plt.title(label='Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show();


#%% marker size
plt.scatter(total_bill, tip, s=1)
plt.show();
plt.scatter(total_bill, tip, s=100)
plt.show();

size_of_table = tips_df['size'].to_numpy()
size_of_table

size_of_table_scaled = size_of_table
size_of_table_scaled

size_of_table_scaled = [ 3*s**2 for s in size_of_table]
size_of_table_scaled

plt.scatter(total_bill, tip, s=size_of_table_scaled)
plt.show();


#%% legend
scatter = plt.scatter(total_bill, tip, s=size_of_table_scaled)
handles, labels = scatter.legend_elements(prop='sizes')
plt.legend(handles, labels)
plt.show();



df
scatter = plt.scatter('wt', 'mpg', s='hp', data=df)
handles, labels = scatter.legend_elements(prop='sizes')
plt.xlabel('wt')
plt.ylabel('mpg')
plt.legend(handles, labels)
plt.show();

