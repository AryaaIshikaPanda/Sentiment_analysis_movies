#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries
# 

# In[20]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# # Preparing the data

# In[9]:


data_review = pd.read_csv(r"C:\Users\Aryaa Ishika\Downloads\archive (1).zip")
data_review


# In[10]:


data_positive = data_review[data_review['sentiment']=='positive'][:9000]
data_negative = data_review[data_review['sentiment']=='negative'][:1000]


# In[11]:


data_review_imb = pd.concat([data_positive,data_negative ])


# # Dealing with Imbalanced Classes

# In[13]:


colors = sns.color_palette('deep')


# In[14]:


plt.figure(figsize=(10,4), tight_layout=True)


# In[17]:


plt.bar(x=['Positive', 'Negative'],
        height=data_review_imb.value_counts(['sentiment']),
        color=colors[:2])
plt.title('Sentiment')
plt.savefig('sentiment.png')
plt.show()


# In[23]:



# Separate the majority and minority classes
data_majority = data_review_imb[data_review_imb.sentiment == data_review_imb.sentiment.value_counts().idxmax()]
data_minority = data_review_imb[data_review_imb.sentiment == data_review_imb.sentiment.value_counts().idxmin()]

# Calculate the number of samples needed to balance the classes
n_samples = data_majority.shape[0] - data_minority.shape[0]

# Randomly sample with replacement from the minority class
data_minority_upsampled = data_minority.sample(n=n_samples, replace=True, random_state=0)

# Combine the majority class with the upsampled minority class
data_review_bal = pd.concat([data_majority, data_minority, data_minority_upsampled])

# Shuffle the combined DataFrame
data_review_bal = data_review_bal.sample(frac=1, random_state=0).reset_index(drop=True)

# Display the balanced DataFrame
print(data_review_bal)


# In[24]:


print(data_review_imb.value_counts('sentiment'))
print(data_review_bal.value_counts('sentiment'))


# # Splitting data into train and test set

# In[25]:


from sklearn.model_selection import train_test_split


# In[27]:


train,test = train_test_split(data_review_bal,test_size =0.30,random_state=40)


# In[28]:


train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']


# In[29]:


train_y.value_counts()


# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[32]:


tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
# also fit the test_x_vector
test_x_vector = tfidf.transform(test_x)


# In[34]:


#Transforming
pd.DataFrame.sparse.from_spmatrix(train_x_vector,
                                  index=train_x.index,
                                  columns=tfidf.get_feature_names_out ())


# # Selecting Model

# In[37]:


#Support_Vector_Machine
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)


# In[38]:


print(svc.predict(tfidf.transform(['Good movie'])))
print(svc.predict(tfidf.transform(['Excellent movie'])))
print(svc.predict(tfidf.transform(['Not good'])))


# In[39]:


#Logistic_Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(train_x_vector,train_y)


# In[40]:


#Classification_Report
from sklearn.metrics import classification_report

print(classification_report(test_y,
                            svc.predict(test_x_vector),
                            labels = ['positive','negative']))


# In[ ]:




