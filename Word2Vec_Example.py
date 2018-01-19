
# coding: utf-8

# In[1]:


import pandas as pd
import csv
import numpy as np


# In[2]:


words = pd.read_table("glove_word2Vec/glove.6B/glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)


# In[3]:


words.shape


# In[4]:


len(words.index)


# In[5]:


def vec(w):
  return words.loc[w].as_matrix()


# In[6]:


vec("shark")


# In[7]:


vec("fish")


# In[8]:


words_matrix = words.as_matrix()

def find_closest_word(v):
    diff = words_matrix - v
    delta = np.sum(diff * diff, axis=1)
    i = np.argmin(delta)
    return words.iloc[i].name

def find_five_closest(v):
    diff = words_matrix - v
    delta = np.sum(diff * diff, axis=1)
    test = np.argsort(delta)
    for x in range(0, 6):
        i = test[x]
        print(words.iloc[i].name)


# In[10]:


find_five_closest(vec("fish"))


# In[11]:


find_five_closest(vec("obama") - vec("usa") + vec("france"))


# In[12]:


find_five_closest(vec("king") - vec("man") + vec("woman"))


# In[13]:


find_five_closest(vec("car") + vec("wings") + vec("fly"))


# In[15]:


find_five_closest(vec("best") - vec("good") + vec("bad"))

