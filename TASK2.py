#!/usr/bin/env python
# coding: utf-8

# # task 2: Prediction using Unsupervised machine learning
# ## from the given 'iris' dataset, predict the optimum number of cluster and represent it visually

# ### importing all the libraries.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans


# In[3]:


iris = pd.read_csv("iris.csv")


# ### data overview

# In[4]:


iris.head()


# In[5]:


iris.isnull().sum()


# In[6]:


iris.info()


# In[7]:


iris.describe()


# In[8]:


iris.shape


# ### Data Preprocessing

# In[9]:


x = iris.drop(['Id', 'Species'], axis=1)
x.head()


# ### K Means Clustering

# In[10]:


x= iris.iloc[:, [0,1,2,3]].values


# In[11]:


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[12]:


wcss=[]

for i in range (1,10):
    kmeans=KMeans(i)
    kmeans.fit(x)
    wcss_iter=kmeans.inertia_
    wcss.append(wcss_iter)
wcss


# ### Plotting a result on to a line graph, allowing us to observe "The elbow".

# In[21]:


nu_clusters = range(1,10)
plt.plot(nu_clusters,wcss, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Clusters Sum of squares')


# ### From the upper graph we came to know that elbow is at 3 so there will be three clusters.

# In[19]:


kmeans = KMeans(n_clusters=3, init= 'k-means++',max_iter=300,n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)


# ### Clusters

# In[15]:


y_kmeans


# ### Visualisation of Clusters

# In[18]:


plt.figure(figsize=(10,6))
plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], s=100, c='red', label='Iris-setosa')
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], s=100, c= 'blue', label='Iris-versicolar')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], s=100,c='green', label='Iris-verginica')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='yellow', label='Centroids')
plt.legend()


# ## From the given dataset we can see the the optimum number of clusters are three and visualised it.

# In[ ]:




