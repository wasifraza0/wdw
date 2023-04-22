
# coding: utf-8

# # Exploratory Data Analysis

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:\\Users\\asus\\Downloads\\haberman.csv") 


# In[3]:


df.head()


# In[4]:


df.sample(10)


# In[5]:


df.shape


# In[6]:


df.info()


# There is no null values present in this dataset

# In[7]:


df.describe()


# In[8]:


print(df.columns)


# Age=total age of the patient \
# Year=when optation of the patient \
# Nodes= number of nodes persent in body \
# Status= status are persent are in form of 1 & 2 
# 
# 

# In[9]:


df["status"].value_counts()


# 1= the patient survived 5 years or longer \
# 2= the patient died within 5 year \
# This is imbalance dataset

# # Objective: - Whether the patient survived more than 5 years or not.
# 
# 

# In[14]:


#2-D scatter plot:
#ALWAYS understand the axis: labels and scale.

df.plot(kind='scatter', x='nodes', y='status');

plt.legend('g')
plt.show()

#cannot make much sense out it. 
#What if we color the points by thier class-label/flower-type.


# In[15]:


#2-D scatter plot:
#ALWAYS understand the axis: labels and scale.

df.plot(kind='scatter', x='age', y='status') ;
plt.show()

#cannot make much sense out it. 
#What if we color the points by thier class-label/flower-type.


# In[16]:


#2-D scatter plot:
#ALWAYS understand the axis: labels and scale.

df.plot(kind='scatter', x='year', y='status') ;
plt.show()

#cannot make much sense out it. 
#What if we color the points by thier class-label/flower-type.


# In[17]:


# 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn. 
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="status", size=4)    .map(plt.scatter, "age", "year")    .add_legend();
plt.show();

# Notice that the blue points can be easily seperated 
# from red and green by drawing a line. 
# But red and green data points cannot be easily seperated.
# Can we draw multiple 2-D scatter plots for each combination of features?
# How many cobinations exist? 4C2 = 6.


# # pairwise scatter plot

# In[18]:


# pairwise scatter plot: Pair-Plot
# Dis-advantages: 
##Can be used when number of features are high.
##Cannot visualize higher dimensional patterns in 3-D and 4-D. 
#Only possible to view 2D patterns.
sns.pairplot(df,hue="status",vars=["age","year","nodes"],size=4)
plt.show()


# Observations
# 
# from above plot we not getting any useful information.\
# all data points are spread across in the status labels.

# # uni-variate analysis

# In[19]:


sns.FacetGrid(df, hue="status", size=3)    .map(sns.distplot, "age")    .add_legend();
plt.show();


# In[20]:


sns.FacetGrid(df, hue="status", size=3)    .map(sns.distplot, "year")    .add_legend();
plt.show();


# In[31]:


sns.FacetGrid(df, hue="status", size=10)    .map(sns.distplot, "nodes")    .add_legend();
plt.show();


# observations
# 
# we not get into a good conclusion from histograms. \
# feature 'age and year' is not relavant for classification of status because there is higher overlapping between class labels.\
# faeture variable 'node' is more relavant for classification.so need to more concentrate on node variable

# # Pdf And Cdf

# In[22]:


df_1 = df.loc[df["status"] == 1]
df_2 = df.loc[df["status"] == 2]


# In[23]:


count,edges=np.histogram(df_1['nodes'],bins=10,density=True)
pdf=count/sum(count)
cdf=np.cumsum(pdf)
print("bin edges",edges[1:])
print(" ")
print("probability density function")
print(" ")
print(pdf)
print(" ")
print("Cumulative distribution function")
print(" ")
print(cdf)
plt.plot(edges[1:],pdf,label="pdf")
plt.plot(edges[1:],cdf,label="cdf")
plt.xlabel("nodes")
plt.title("Pdf and Cdf of status = 1")
plt.legend()
plt.show()


# observations
# 
# about 92% patients who survived have postive node between 0 to 10\
# only 3-4% of pattients survived have nodes greater than 15

# In[24]:


count,edges=np.histogram(df_2['nodes'],bins=10,density=True)
pdf=count/sum(count)
cdf=np.cumsum(pdf)
print("bin edges",edges[1:])
print(" ")
print("probability density function")
print(" ")
print(pdf)
print(" ")
print("Cumulative distribution function")
print(" ")
print(cdf)
plt.plot(edges[1:],pdf,label="pdf")
plt.plot(edges[1:],cdf,label="cdf")
plt.xlabel("nodes")
plt.title("Pdf and Cdf of status = 2")
plt.legend()
plt.show()


# # observations
# 
# about 72% patients who not survived have 0 to 10 nodes\
# patients who not survived contains more number of nodes

# # Box plot

# In[25]:


plt.figure(figsize=(7,5))
plt.title("status vs nodes")
sns.boxplot(data=df,x='status',y='nodes',hue='status')
plt.show()


# In[26]:


plt.figure(figsize=(7,5))
plt.title("Box Plot (status vs age)")
sns.boxplot(data=df,x='status',y='age',hue='status')
plt.show()


# In[27]:


plt.figure(figsize=(7,5))
plt.title("Box Plot (status vs year)")
sns.boxplot(data=df,x='status',y='year',hue='status')
plt.show()


# observations
# 
# about 50% of patients who survived(status =1) have no positive nodes\
# large number of outlier is present even if positive node is high some patients are survived

# # Violin Plot

# In[28]:


plt.figure(figsize=(7,5))
plt.title("Violin Plot (status vs nodes)")
sns.violinplot(data=df,x='status',y='nodes',hue='status')
plt.show()


# In[29]:


plt.figure(figsize=(7,5))
plt.title("Violin Plot (status vs age)")
sns.violinplot(data=df,x='status',y='age',hue='status')
plt.show()


# In[30]:


plt.figure(figsize=(7,5))
plt.title("Violin Plot (status vs year)")
sns.violinplot(data=df,x='status',y='year',hue='status')
plt.show()


# # Box Plot & Violin Plot
# --> if we see the box plot between nodes and Survival Status , there Seems to be to many Outliers, especially for long Survival Status. it is always good to remove these Outliers for better analysis
# 
# --> about 50% of the Patient who survived has no postive axillay nodes
# 
# --> if you look at the Box-plot of Non_Surivval Patient (less than 5 years). the 50% percentile and 75% gap seems almose thrice as gap between 25% to 50%, in other way about 50% of Patient who Survived less has Axillary nodes of 4 or less and other 25% Patient has axillary nodes from 4-11. and once it crossess 11, the chances of Survival are very very less

# # Conclusion
# -->with this Data it is not possbile to analyse the effect of parameters i,e Age, Operation Year and Axillary nodes on Survival Statu
# 
# -->if we look at the cdf and pdf Analysis we find that 75% of the Patient who survived for 5 years or longer has axil nodes of 4 or less and 90% of Patient whos survived has axil nodes of 9 or less
# 
# --> none of the variable either single or pair can help in categorizing the survivial status using if else condition
# 
# --> if there would have been large quantity of data, the analysis might have been different.

# 
# 
