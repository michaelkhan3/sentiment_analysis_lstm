
# coding: utf-8

# In[1]:


import pandas as pd
import csv
import numpy as np
import string
import re


# In[2]:


words = pd.read_table("glove_word2Vec/glove.6B/glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)


# In[3]:


wordsVector = words.as_matrix().astype(np.float32)
wordsList = words.index.tolist()


# In[5]:


print(wordsVector.shape)
print(len(wordsList))


# In[8]:


baseballIndex = wordsList.index('baseball')
wordsVector[baseballIndex]


# In[9]:


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

# In[10]:


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


# In[11]:


testSent = "Hello, how are you doing today?"

testSentVec = convert_sentence(testSent)

print(testSentVec)


# In[12]:


with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordsVector, firstSentence).eval().shape)


# ## Load Data
# 
# Now we will load the movie review data. 
# The data comes from https://www.kaggle.com/c/word2vec-nlp-tutorial/data

# In[13]:


train_reviews = pd.read_table("movie_review_dataset/labeledTrainData/labeledTrainData.tsv", sep='\t')


# In[14]:


train_reviews.head()


# In[15]:


train_reviews.shape


# ## Exploratory analysis
# 
# Exploring the number of words in each review 

# In[16]:


def sentence_len(sentence):
    sentence = sentence.translate(string.punctuation)
    sentence = sentence.split(" ")
    return len(sentence)


# In[17]:


num_words = train_reviews.apply(lambda row: sentence_len(row['review']), axis=1)


# In[18]:


num_words.describe()


# In[19]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist(num_words, 50)
plt.xlabel("Sentence Length")
plt.ylabel("Frequency")
plt.axis([0, 1200, 0, 8000])
plt.show()


# ## Converting words to word vecs

# Code in this section has been commented out as it takes quite alot of time to run it.
# Instead we can just load data from a csv file

# #### TODO
# 
# We may have to have all reviews be of the same length. This is kind of inconvinient since we will lose information.
# Check if anyone has a solution to this.
# 
# ##### UPDATE 6/12/2017
# Now that we end up with a list of lists from this conversion, it may be possible to have the internal lists be of different length. Because I am converting the list to a DataFrame I wont try to have different lengths yet.

# First, try this with a subset of the training data. Maybe the first 500 rows.

# In[20]:


# subset_reviews = train_reviews.iloc[0:500]


# Using apply wasn't working correctly so I decided to use a list comprehension

# In[27]:


# subset_reviews_ids = [convert_sentence(row[3]) for row in subset_reviews.itertuples()]


# In[28]:


# subset_reviews_ids = pd.DataFrame(subset_reviews_ids)
# subset_reviews_ids.shape


# In[29]:


# subset_reviews_ids.head(2)


# Now apply it to whole dataset. 
# 
# This takes a **LONG** time to run therefore, I have saved the output file as a csv which can loaded to skip this step.

# In[30]:


# train_reviews_ids = [convert_sentence(row[3]) for row in train_reviews.itertuples()]


# In[31]:


# train_reviews_ids_df = pd.DataFrame(train_reviews_ids)


# In[32]:


# train_reviews_ids_df.shape


# In[33]:


# train_reviews_ids_df.head()


# In[34]:


# train_reviews_ids_df.to_csv("movie_review_dataset/labeledTrainData/ids_matrix.csv")


# In[20]:


train_reviews_ids_df = pd.read_csv("movie_review_dataset/labeledTrainData/ids_matrix.csv")


# ## RNN Model

# Setting hyper parameters

# In[21]:


batch_size = 24
lstm_units = 64
num_classes = 2
itterations = 100000


# In[22]:


import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, num_classes])
input_data = tf.placeholder(tf.int32, [batch_size, maxSeqLength])


# In[24]:


data = tf.Variable(tf.zeros([batch_size, maxSeqLength, numDimensions]), dtype=tf.float32)

data = tf.nn.embedding_lookup(wordsVector, input_data)


# In[25]:


lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)


# In[26]:


weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)


# In[27]:


correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


# In[28]:


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


# In[29]:


import datetime

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)


# In[ ]:




