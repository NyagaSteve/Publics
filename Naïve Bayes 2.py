#!/usr/bin/env python
# coding: utf-8

# In[222]:


import nltk
import numpy as np
import pandas as pd
import re
import string
import os
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


# In[221]:


import pandas as pd
data = pd.read_csv('US Airline.csv')
#data = pd.read_csv('Tweets.csv')


# In[6]:


data.head(100)


# In[43]:


data.columns


# In[44]:


import matplotlib.pyplot as plt
counts = data['airline_sentiment'].value_counts()
counts.plot(kind='bar')

plt.title('Number of Tweets per Class')
plt.xlabel('Sentiment Class')
plt.ylabel('Number of Tweets')
plt.show()


# In[45]:


import matplotlib.pyplot as plt
grouped = data.groupby(['airline', 'airline_sentiment']).size().unstack()

fig = plt.figure(figsize = (10,20))
grouped.plot(kind='bar', stacked=True)
plt.title('Airline Sentiment by Airline')
plt.xlabel('Airline')
plt.ylabel('Number of Tweets')
plt.show()


# In[46]:


neg_reason_counts = data['negativereason'].value_counts()

plt.pie(neg_reason_counts, labels=neg_reason_counts.index)
plt.title('Negative Reasons for Tweets')
plt.show()


# In[47]:


# Separate negative and positive sentiment tweets
neg_data = data[data['airline_sentiment'] == 'negative']
pos_data = data[data['airline_sentiment'] == 'positive']

# Sample an equal number of negative and positive tweets
num_samples = min(len(neg_data), len(pos_data))
neg_data = neg_data.sample(n=num_samples, random_state=42)
pos_data = pos_data.sample(n=num_samples, random_state=42)

# Concatenate the negative and positive sentiment tweets
balanced_data = pd.concat([neg_data, pos_data])

# Shuffle the rows
balanced_data = balanced_data.sample(frac=1, random_state=42)


# In[18]:


counts = balanced_data['airline_sentiment'].value_counts()
counts.plot(kind='bar')

plt.title('Number of Tweets per Class')
plt.xlabel('Sentiment Class')
plt.ylabel('Number of Tweets')
plt.show()


# In[19]:


print('Number of positive sentiment tweets: {}'.format(len(pos_data)))
print('Number of negative sentiment tweets: {}'.format(len(neg_data)))


# In[21]:


from sklearn.model_selection import train_test_split

# Split into features and target
X = balanced_data['text'].values.tolist()
y = balanced_data['airline_sentiment'].values.tolist()

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state = 42)


# In[23]:


# Map positive to 1 and negative to 0
y_train = [int(sent == 'positive') for sent in y_train]
y_test = [int(sent == 'positive') for sent in y_test]


# In[66]:


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet (str): a string containing a tweet
    Output:
        tweets_clean (list): a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks    
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


# In[69]:


sample_tweet = X_train[0]
print('Tweet before preprocessing: {}'.format(sample_tweet))
#print('Tweet after preprocessing: {}'.format(process_tweet(sample_tweet)))


# In[ ]:


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets (list): a list of tweets
        ys (np.array): an m x 1 array with the sentiment label of each tweet (either 0 or 1)
    Output:
        freqs (Dictionary): a dictionary mapping each (word, sentiment) pair to its frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs


# In[ ]:


freqs = build_freqs(X_train, y_train)


# In[29]:


T = len(X_train)
T_pos = y_train.count(1)
T_neg = y_train.count(0)

print('Total number of tweets: {}'.format(T))
print('Number of positive tweets: {}'.format(T_pos))
print('Number of negative tweets: {}'.format(T_neg))


# In[49]:


# Calculate the probabilities for each class
Prob_T_pos = T_pos/T
Prob_T_neg = T_neg/T

print('Probability of positive class: {}'.format(Prob_T_pos))
print('Probability of negative class: {}'.format(Prob_T_neg))


# In[58]:


import nltk 
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import word_tokenize,sent_tokenize
# Removing stop words using nltk lib

#Tokenization of text
tokenizer=ToktokTokenizer() 

#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

#Removing standard english stopwords like prepositions, adverbs
stop = set(stopwords.words('english'))
print("NLTK stop word lists \n")
print(stop)

#Removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


# In[73]:


df_clean = pd.DataFrame({'clean'})
df_clean = df_clean.dropna().drop_duplicates()
print(df_clean.shape)
df_clean.head(5)


# In[ ]:


train = pd.read_csv('US Airline.csv')


# In[115]:


length_train_dataset = train['negativereason'].str.len()
#length_test_dataset = test['tweet'].str.len()
plt.hist(length_train_dataset, bins=20,label="negativereason")
#plt.hist(length_test_dataset, bins=20,label="negativereason")
plt.legend() 
plt.show()


# In[122]:


combine=train.append(train,ignore_index=True) #train and test dataset are combined
combine.shape


# In[128]:


combine.head()


# In[138]:


combine['tidy_text'] = np.vectorize(remove_pattern)(combine['text'],"@[\w]*") 
combine.head()


# In[139]:


combine['tidy_text'] = combine['tidy_text'].str.replace("[^a-zA-Z#]"," ")
combine.head(10)


# In[ ]:


combine['tidy_text'] = combine['tidy_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) #removing words whose length is less than 3


# In[140]:


combine.head()


# In[141]:


tokenized_tweet = combine['tidy_text'].apply(lambda x:x.split()) #it will split all words by whitespace
tokenized_tweet.head()


# In[214]:


data[data['airline_sentiment']=="negative"]["text"]


# In[ ]:




