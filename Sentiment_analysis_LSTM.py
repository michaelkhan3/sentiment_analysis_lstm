
# coding: utf-8

# In[ ]:


import pandas as pd
import csv
import numpy as np
import string
import re


# In[ ]:


words = pd.read_table("glove_word2Vec/glove.6B/glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)


# In[ ]:


wordsVector = words.as_matrix()
wordsList = words.index.tolist()


# In[ ]:


print(wordsVector.shape)
print(len(wordsList))


# In[ ]:


type(wordsList)


# In[ ]:


baseballIndex = wordsList.index('baseball')
wordsVector[baseballIndex]


# In[ ]:


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

# In[ ]:


def convert_sentence(sentence):
    # Got from analysis below.
    maxSeqLength = 250
    index_count = 0
    
    remove_special_chars = re.compile("[^A-Za-z0-9 ]+")
    
    sentence = sentence.lower()
    sentence = sentence.translate(string.punctuation)
    sentence = re.sub(remove_special_chars, "", sentence)
    sentence = sentence.split(" ")
    if len(sentence) > maxSeqLength:
        sentence = sentence[:maxSeqLength]
    sentenceList = np.zeros((maxSeqLength), dtype='int32')
    for word in sentence:
        try:
            sentenceList[index_count] = wordsList.index(word)
        except ValueError:
            # TODO create a vector for unknow words
            # https://groups.google.com/forum/#!topic/globalvectors/n6BYywiENGo
            # For now just skip unkown words
            sentenceList[index_count] = 0
            
        index_count = index_count + 1
        
    return np.array(sentenceList)


# In[ ]:


testSent = "Hello, how are you doing today?"

testSentVec = convert_sentence(testSent)

print(testSentVec)


# In[ ]:


with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordsVector, firstSentence).eval().shape)


# ## Load Data
# 
# Now we will load the movie review data. 
# The data comes from https://www.kaggle.com/c/word2vec-nlp-tutorial/data

# In[ ]:


train_reviews = pd.read_table("movie_review_dataset/labeledTrainData/labeledTrainData.tsv", sep='\t')


# In[ ]:


train_reviews.head()


# In[ ]:


train_reviews.shape


# ## Exploratory analysis
# 
# Exploring the number of words in each review 

# In[ ]:


def sentence_len(sentence):
    sentence = sentence.translate(string.punctuation)
    sentence = sentence.split(" ")
    return len(sentence)


# In[ ]:


num_words = train_reviews.apply(lambda row: sentence_len(row['review']), axis=1)


# In[ ]:


num_words.describe()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

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
# 
# ##### UPDATE 6/12/2017
# Now that we end up with a list of lists from this conversion, it may be possible to have the internal lists be of different length. Because I am converting the list to a DataFrame I wont try to have different lengths yet.

# First, try this with a subset of the training data. Maybe the first 500 rows.

# In[ ]:


subset_reviews = train_reviews.iloc[0:500]


# In[ ]:


#subset_reviews_ids = subset_reviews.apply(lambda row: convert_sentence(row['review']), axis=1)


# In[ ]:


subset_reviews_ids = [convert_sentence(row['review']) for index, row in subset_reviews.iterrows()]


# In[ ]:


subset_reviews_ids = pd.DataFrame(subset_reviews_ids)
subset_reviews_ids.shape


# In[ ]:


subset_reviews_ids.head(2)


# In[ ]:


train_reviews_ids = train_reviews.apply(lambda row: convert_sentence(row['review']), axis=1)

