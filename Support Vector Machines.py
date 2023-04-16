#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[18]:


import pandas as pd
#df = pd.read_csv('US AIRLINE.csv')
df = pd.read_csv('US AIRLINE.csv')
df.head(100)


# In[14]:


X1=df['tweet_id']
X2=df['airline_sentiment_confidence']
X_training=np.array(list(zip(X1,X2)))
X_training


# In[15]:


y_training=df['airline_sentiment_confidence']
y_training


# In[19]:


target_names=['-1','+1']
target_names


# In[20]:


idxPlus=y_training[y_training<0].index
idxMin=y_training[y_training>0].index
plt.scatter(X_training[idxPlus,0],X_training[idxPlus,1],c='b',s=50)
plt.scatter(X_training[idxMin,0],X_training[idxMin,1],c='r',s=50)
plt.legend(target_names,loc=2)
plt.xlabel('tweet_id')
plt.ylabel('airline_sentiment_confidence');
plt.savefig('chart0.png')


# In[29]:


X = df.drop('retweet_count', axis=1)
y = df['retweet_count']


# In[36]:


df.isna().sum()


# In[37]:


df.describe()


# In[38]:


df.info()   


# In[19]:


col_names = df.columns

col_names


# In[31]:


import seaborn as sns
from sklearn.datasets import load_iris


# In[32]:


iris = load_iris()


# In[33]:


print(iris.keys())


# In[46]:


X=df.iloc[:, :-1]
X.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




