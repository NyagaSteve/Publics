#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[ ]:


# load the dataset
amazon_df = pd.read_csv('US AIRLINE.csv')


# In[ ]:


# drop the irrelevant columns and missing values
amazon_df = amazon_df[['Reviews', 'Rating']].dropna()


# In[ ]:


# convert the rating to a binary sentiment (positive or negative)
amazon_df['Sentiment'] = np.where(amazon_df['Rating'] > 3, 'positive', 'negative')


# In[ ]:


# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(amazon_df['Reviews'], amazon_df['Sentiment'], test_size=0.2, random_state=42)


# In[ ]:


# create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()


# In[ ]:


# fit and transform the training data
X_train = vectorizer.fit_transform(X_train


# In[ ]:


# transform the testing data
X_test = vectorizer.transform(X_test)


# In[ ]:


# Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)
nb_prec = precision_score(y_test, nb_pred, pos_label='positive')
nb_rec = recall_score(y_test, nb_pred, pos_label='positive')


# In[ ]:


# SVM model
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
svm_prec = precision_score(y_test, svm_pred, pos_label='positive')
svm_rec = recall_score(y_test, svm_pred, pos_label='positive')


# In[ ]:


# Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred, pos_label='positive')
lr_rec = recall_score(y_test, lr_pred, pos_label='positive')


# In[ ]:


# print the performance metrics
print('Naive Bayes: Accuracy =', nb_acc, 'Precision =', nb_prec, 'Recall =', nb_rec)
print('SVM: Accuracy =', svm_acc, 'Precision =', svm_prec, 'Recall =', svm_rec)
print('Logistic Regression: Accuracy =', lr_acc, 'Precision =', lr_prec, 'Recall =', lr_rec)


# In[ ]:





# In[ ]:




