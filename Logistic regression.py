#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.simplefilter(action='ignore')


# In[2]:


train_df = pd.read_csv("US AIRLINE.csv")
train_df.head()


# In[3]:


print('The number of samples into the train data is {}.'.format(train_df.shape[0]))


# In[4]:


test_df = pd.read_csv("US AIRLINE.csv")
test_df.head()


# In[5]:


train_df.isnull().sum()


# In[8]:


print('Percent of missing airline_sentiment_confidence records is %.2f%%' %((train_df['airline_sentiment_confidence'].isnull().sum()/train_df.shape[0])*100))


# In[13]:


ax = train_df["airline_sentiment_confidence"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df["airline_sentiment_confidence"].plot(kind='density', color='teal')
ax.set(xlabel='airline_sentiment_confidence')
plt.xlim(-10,85)
plt.show()


# In[15]:


print('Boarded passengers grouped by port of embarkation (U = UNITED, US = US Airways, A = America):')
print(train_df['airline'].value_counts())
sns.countplot(x='airline', data=train_df, palette='Set2')
plt.show()


# In[16]:


train_df.isnull().sum()


# In[53]:


plt.figure(figsize=(15,8))
#ax = sns.kdeplot(final_train["airline_sentiment_confidence"][final_train.Survived == 1], color="darkturquoise", shade=True)
#sns.kdeplot(final_train["airline_sentiment_confidence"][final_train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Positive', 'Negative'])
plt.title('Density Plot of airline_sentiment_confidence for positive and negative')
ax.set(xlabel='airline_sentiment_confidence')
plt.xlim(-10,85)
plt.show()


# In[54]:


train_df['negativereason'].unique()


# In[57]:


#Data exploration
train_df['airline_sentiment_confidence'].value_counts()


# In[60]:


sns.countplot(x='airline_sentiment',data=train_df, palette='hls')
plt.show()
plt.savefig('count_plot')


# In[63]:


train_df.groupby('user_timezone').mean()


# In[ ]:


#Converting Categorical Features
train_df.info()


# In[ ]:


#Virtualization

get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(train_df.airline,train_df.name).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('airline')
plt.ylabel('name')
plt.savefig('purchase_fre_job')


# In[ ]:


#Train Test Split
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('airline_sentiment_confidence',axis=1), 
                                                    train['airline_sentiment_confidence'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:


#Training and Predicting
from sklearn.linear_model import LogisticRegression


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


#Evaluation
from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[ ]:




