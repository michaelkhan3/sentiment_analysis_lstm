
# coding: utf-8

# In[22]:

import pandas as pd
import csv
import numpy as np
import string
import re


# In[3]:

words = pd.read_table("glove_word2Vec/glove.6B/glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)


# In[4]:

wordsVector = words.as_matrix()
wordsList = words.index.tolist()


# In[5]:

print(wordsVector.shape)
print(len(wordsList))


# In[6]:

type(wordsList)


# In[7]:

baseballIndex = wordsList.index('baseball')
wordsVector[baseballIndex]


# In[8]:

import tensorflow as tf

maxSeqLength = 10
numDimensions = 300

firstSentence = np.zeros((maxSeqLength), dtype='int32')
firstSentence[0] = wordsList.index("i")
firstSentence[1] = wordsList.index("tought")
firstSentence[2] = wordsList.index("the")
firstSentence[3] = wordsList.index("movie")
firstSentence[4] = wordsList.index("was")
firstSentence[5] = wordsList.index("incredible")
firstSentence[6] = wordsList.index("and")
firstSentence[7] = wordsList.index("inspiring")
print(firstSentence.shape)
print(firstSentence)


# ### Converting sentences
# 
#  Creating a utility function to convert sentences into an numpy array of words.

# In[27]:

def convert_sentence(sentence):
    remove_special_chars = re.compile("[^A-Za-z0-9 ]+")
    
    sentence = sentence.lower()
    sentence = sentence.translate(None, string.punctuation)
    sentence = re.sub(remove_special_chars, "", sentence)
    sentence = sentence.split(" ")
#     if len(sentence) > maxSeqLength:
#         sentence = sentence[:maxSeqLength]
    sentenceList = []
    for word in sentence:
        try:
            sentenceList.append(wordsList.index(word))
        except ValueError:
            # TODO create a vector for unknow words
            # https://groups.google.com/forum/#!topic/globalvectors/n6BYywiENGo
            # For now just skip unkown words
            pass
        
    return np.array(sentenceList)


# In[24]:

testSent = "Hello, how are you doing today?"

testSentVec = convert_sentence(testSent)

print(testSentVec)


# In[11]:

with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordsVector, firstSentence).eval().shape)


# ## Load Data
# 
# Now we will load the movie review data. 
# The data comes from https://www.kaggle.com/c/word2vec-nlp-tutorial/data

# In[12]:

train_reviews = pd.read_table("movie_review_dataset/labeledTrainData/labeledTrainData.tsv", sep='\t')


# In[13]:

train_reviews.head()


# ## Exploratory analysis
# 
# Exploring the number of words in each review 

# In[16]:

def sentence_len(sentence):
    sentence = sentence.translate(None, string.punctuation)
    sentence = sentence.split(" ")
    return len(sentence)


# In[18]:

num_words = train_reviews.apply(lambda row: sentence_len(row['review']), axis=1)


# In[21]:

num_words.describe()


# In[20]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

plt.hist(num_words, 50)
plt.xlabel("Sentence Length")
plt.ylabel("Frequency")
plt.axis([0, 1200, 0, 8000])
plt.show()


# ## Converting words to word vecs

# #### TODO
# 
# We may have to have all reviews be of the same length. This is kind of inconvinient since we will lose information.
# Check if anyone has a solution to this.

# In[ ]:

train_reviews_ids = train_reviews.apply(lambda row: convert_sentence(row['review']), axis=1)


# In[ ]:



