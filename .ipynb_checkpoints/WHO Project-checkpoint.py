#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import statsmodels.api as sm
import statsmodels.tools


# In[46]:


df = pd.read_csv("C:\\Users\\Toby\\Documents\\Digital Futures\\Projects\\WHO Project\\Life Expectancy Data.csv")
df.shape


# In[48]:


feature_cols = list(df.columns)
feature_cols.remove('Life_expectancy')

X = df[feature_cols]
y = df['Life_expectancy']


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
X_train.head()


# In[52]:


print(f'Same number of records in Train: {X_train.shape[0] == y_train.shape[0]}')
print(f'Same number of records in Test: {X_test.shape[0] == y_test.shape[0]}')


# In[54]:


assert(all(X_train.index == y_train.index)), "There is some index mismatch in Train"
assert(all(X_test.index == y_test.index)), "There is some index mismatch in Test"

