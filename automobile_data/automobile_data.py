
# coding: utf-8

# ### Exploratory data analysis on Automobile data set

# In[1]:


import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# Reading the data from the file
data = pd.read_csv('data/Automobile_data.csv')
data.head()


# In[3]:


# Displaying the statistics of the data
data.describe()


# In[4]:


# Checking for null values in the data set
data.isnull().sum()


# In[5]:


# Checking for the values for ? in the data
for i,j in zip(data.columns, (data.values.astype(str) == '?') .sum(axis = 0)):
    if j > 0:
        print('{} : {} records'.format(i, j))


# In[6]:


# Set the missing values for normalized - losses, horsepower, price with the mean value
#data.mean()
for i in data.columns:
    temp = data[i].loc[data[i].astype(str) == '?'].count()
    print('{} : {} records'.format(i, temp))


# In[7]:


temp = data['normalized-losses'].loc[data['normalized-losses'].astype(str) != '?']
mean = temp.astype(str).astype(int).mean()
#mean
data['normalized-losses'] = data['normalized-losses'].replace(to_replace = '?', value = mean).astype(int)
data['normalized-losses'].head()


# In[8]:


# Checking the value for price
data['price'].str.isnumeric().value_counts()


# In[9]:


# Set the value of non-numeric with mean values
temp_price = data['price'].loc[data['price'].astype(str) != '?']
#print(temp_price)
mean_price = temp_price.astype(str).astype(int).mean()
#mean_price
data['price'] = data['price'].replace(to_replace = '?', value = mean).astype(int)


# In[10]:


# Checking values for horse power which are non-numeric
data['horsepower'].str.isnumeric().value_counts()


# In[11]:


#data.mean()
for i in data.columns:
    #print(data[i])
    if i not in ['make', 'fuel-type', 'aspiration', 'num-of-doors','engine-location', 'wheel-base', 'stroke', 'bore','num-of-cylinders', 'length','width','height', 'body-style','engine-type', 'body-style', 'drive-wheels', 'engine', 'fuel-system', 'compression-ratio']:
        temp = data[i].loc[data[i].astype(str) != '?']
        mean = temp.astype(str).astype(int).mean()
        data[i] = data[i].replace(to_replace = '?', value = mean).astype(int)


# In[12]:


data.head()


# In[13]:


# Outlier detection using z-score for horse power
data[np.abs(data['horsepower'] - data['horsepower'].mean()) <= (3*data['horsepower'].std())]
data.head(5)


# In[14]:


# Cross check again for ? values
for i in data.columns:
    temp = data[i].loc[data[i].astype(str) == '?'].count()
    print('{} : {} records'.format(i, temp))


# In[15]:


#data['num-of-doors'] = pd.to_numeric(data['num-of-doors'],errors='coerce')
for i in data.columns:
    temp = data[i].loc[data[i].astype(str) == '?'].count()
    print('{} : {} records'.format(i, temp))
    if temp > 0:
        data[i] = pd.to_numeric(data[i], errors = 'coerce')


# In[29]:





# ### Plot Number of Vehicles by make

# In[17]:


data.make.value_counts()


# In[18]:


data.make.value_counts().plot(kind = 'bar', figsize = (10,5))
plt.title('Make vs Number of Vehicles')
plt.xlabel('Make')
plt.ylabel('Number of Vehicles')


# ### Insurance risk ratings

# In[19]:


data.symboling.hist(bins = 5)
plt.xlabel('Risk Rating')
plt.ylabel('Number of Vehicles')


# ### Fuel type bar chart and Pie chart

# In[20]:


# Bar Chart
data['fuel-type'].value_counts().plot(kind = 'bar')
plt.title('Fuel Type vs Count')
plt.xlabel('Fuel Type')
plt.ylabel('Count')


# In[21]:


# Pie Chart
data['fuel-type'].value_counts().plot.pie(figsize = (5,5))
plt.title("Fuel type pie diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('Fuel type')


# ### Correlation Analysis

# In[46]:


# We shall plot the correlation analysis graph for better understanding of the correlation between the variables
# in the data set
cor = data.corr()
sns.set_context('notebook', font_scale = 1.0, rc={'line.linewidth':2.5})
plt.figure(figsize=(13,7))
heatmap = sns.heatmap(cor, annot = True, fmt = '.3f')


# ### Box plot for Automobiles

# In[55]:


plt.figure(figsize=(20,10))
sns.boxplot(x = 'make', y = 'price', data = data)


# In[56]:


## Regression model
sns.lmplot('price', 'engine-size', data = data)


# ### Scatter Plot

# In[68]:


plt.scatter(data['engine-size'], data['peak-rpm'])
plt.title('Scatter plot for Engine-SIze vs Peak-rpm')


# ### Bar chart for drive- wheels vs city-mpg

# In[66]:


data.groupby('drive-wheels')['city-mpg'].mean().plot(kind = 'bar')
plt.title('Drive-Wheels vs City-MPG')
plt.xlabel('Drive-Wheels')
plt.ylabel('City-mpg')

