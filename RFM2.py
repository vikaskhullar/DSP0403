Online# -*- coding: utf-8 -*-
#import modules
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime as dt

data = pd.read_csv('OnlineRetail.csv', encoding= 'unicode_escape')
data.dtypes

data = pd.read_csv('OnlineRetail.csv', parse_dates=['InvoiceDate'], encoding= 'unicode_escape')
data.dtypes

pd.set_option('display.max_columns', None)

data.head(5)

pd.set_option('display.max_rows', None)
data.head(10)


data= data[pd.notnull(data['CustomerID'])]


filtered_data=data[['Country','CustomerID']].drop_duplicates()

filtered_data.head(10)

#Top ten country's customer
filtered_data.Country.value_counts().plot(kind='bar')

filtered_data.Country.value_counts().sort_values(ascending=False).head(5)
filtered_data.head(10)



uk_data=data[data.Country=='United Kingdom']

uk_data.info()

uk_data.columns

uk_data.describe()


uk_data = uk_data[(uk_data['Quantity']>0)]


uk_data.info()


uk_data=uk_data[['CustomerID','InvoiceDate','InvoiceNo','Quantity','UnitPrice']]



uk_data['TotalPrice'] = uk_data['Quantity'] * uk_data['UnitPrice']

uk_data.head(10)


uk_data['InvoiceDate'].min(),uk_data['InvoiceDate'].max()

PRESENT = dt.datetime(2011,12,10)




uk_data['InvoiceDate'].dtype
uk_data['InvoiceDate'] = pd.to_datetime(uk_data['InvoiceDate'])
uk_data['InvoiceDate'].dtype

uk_data.head()
uk_data.columns



'''
RFM Analysis

Here, you are going to perform following opertaions:

    For Recency, Calculate the number of days between present date and date of last purchase each customer.
    For Frequency, Calculate the number of orders for each customer.
    For Monetary, Calculate sum of purchase price for each customer.

'''

uk_data.columns

#uk_data[uk_data['CustomerID'] == 541891]

rfm= uk_data.groupby('CustomerID').agg({'InvoiceDate': lambda date: (PRESENT - date.max()).days,
                                        'InvoiceNo': lambda num: len(num),
                                        'TotalPrice': lambda price: price.sum()})

rfm.columns
rfm.head(10)

# Change the name of columns
rfm.columns=['recency','frequency','monetary']
rfm.columns
rfm.head()


rfm['recency'] = rfm['recency'].astype(int)

rfm.head()
rfm.dtypes



rfm['r_quartile'] = pd.qcut(rfm['recency'], 3, ['1','2','3'])
rfm['f_quartile'] = pd.qcut(rfm['frequency'], 3, ['3','2','1'])
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 3, ['3','2','1'])


rfm.head()


rfm[['recency','r_quartile']].sort_values('r_quartile').head(10)

rfm['r_quartile'].unique()

rfm.head(2)

rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)

rfm.head()

rfm = rfm[['RFM_Score']]

rfm.head()

rfm.sort_values('RFM_Score').head(10)

rfm[rfm['RFM_Score'] == '331']



# Filter out Top/Best cusotmers
rfm1=rfm[rfm['RFM_Score']=='111'].sort_values('monetary', ascending=False)

rfm.tail(10)

rfm.to_csv('RFM1.csv')
