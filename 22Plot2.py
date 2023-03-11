#Box Plot

import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_1 = np.random.normal(100, 10, 200)

plt.hist(data_1)

data_1.min()
data_1.max()
data_1.mean()
data_1.std()

data_2 = np.random.normal(90, 20, 200)
data_2.mean()
 
data_3 = np.random.normal(80, 30, 200) 
data_3.mean()

data_4 = np.random.normal(70, 40, 200) 
data_4.mean()

data_5 = np.random.randint(50, 100, 10)
data_5.mean()


data_7 = np.random.normal(70, 10, 11) 
data_7.mean()
data_7.sort()
data_7

data_6 = np.array([1,2,3,8,9])
data_6.mean()
data_6.sort()
data_6





data = [data_1, data_2, data_3, data_4, ] 

fig = plt.figure(figsize =(5, 5), dpi=300, facecolor='r') 
ax = fig.add_axes([1,2,3,4]) 
ax.boxplot(data)
#ax.set_xticklabels(fontsize=14)
ax.tick_params(axis='y', labelsize=30)

# show plot 
plt.show() 

'''
The given code includes a list of heights for various basketball players. 
You need to calculate and output how many players are in the range of 
one standard deviation from the mean                                                                     this is the actual question
the range of one standard deviation from the mean?
'''


x1 = [180, 172, 178, 185, 190, 195, 192, 200, 210, 190]
fig = plt.figure(figsize =(5, 5), dpi=300, facecolor='r') 
ax = fig.add_axes([1,2,3,4]) 
bp = ax.boxplot(x1)

np.mean(x1)
np.median(x1)
np.std(x1)
np.min(x1)
np.max(x1)


mx= np.mean(x1) + np.std(x1)
mi = np.mean(x1) - np.std(x1)

cnt=0
for i in x1:
    if (i>mi and i<mx):
        cnt=cnt+1

cnt
 













   
    
    

import seaborn as sns
sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")
tips.columns
ax = sns.boxplot(x=tips["total_bill"])
ax = sns.boxplot(x=tips["tip"])
sns.scatterplot(tips["total_bill"], tips["tip"])
l1 = [1,4,7,9,10]
np.mean(l1)




#Different dataset
import pandas as pd
data = pd.read_csv('brain_size.csv', sep=';', na_values='.')
data.head()
# Box plot of FSIQ and PIQ (different measures od IQ)
plt.figure(figsize=(4, 3), dpi=300, facecolor= 'Yellow', edgecolor='Red')
data.columns

data.boxplot(['FSIQ', 'PIQ', 'VIQ'])

plt.show();

data
data['FSIQ'] - data['PIQ']
# Boxplot of the difference
plt.figure(figsize=(4, 3))
plt.boxplot(data['FSIQ'] - data['PIQ'])

plt.xticks((0, 1, 2), ('VV', 'FSIQ - PIQ', 'AA' ))

plt.yticks((-10, -15,), ("BB", "AA",))

plt.show();


