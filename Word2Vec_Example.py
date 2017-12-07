
# coding: utf-8

# In[19]:

import pandas as pd
import csv
import numpy as np


# In[2]:

words = pd.read_table("glove_word2Vec/glove.6B/glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)


# In[20]:

words.shape


# In[22]:

len(words.index)


# In[4]:

def vec(w):
  return words.loc[w].as_matrix()


# In[5]:

vec("shark")


# In[6]:

vec("fish")


# In[7]:

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


# In[8]:

find_closest_word(vec("fish"))


# In[9]:

find_five_closest(vec("fish"))


# In[10]:

find_five_closest(vec("obama") - vec("usa") + vec("france"))


# In[11]:

find_five_closest(vec("fish") - vec("water"))


# In[12]:

find_five_closest(vec("king") - vec("man") + vec("woman"))


# In[14]:

find_five_closest(vec("sealion") - vec("water"))


# In[15]:

find_five_closest(vec("michael") - vec("man") + vec("woman"))


# In[16]:

find_five_closest(vec("car") + vec("wings") + vec("fly"))


# In[17]:

find_five_closest(vec("dehydration") + vec("water"))


# In[18]:

find_five_closest(vec("duck") - vec("quack") + vec("meow"))


# In[ ]:



