
# coding: utf-8

# In[1]:


import pandas as pd
import csv
import numpy as np
import string
import re
import tensorflow as tf


# In[2]:


wordsVector = np.load('glove_word2Vec/wordVectors.npy')
wordsList = np.load('glove_word2Vec/wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]


# In[3]:


print(wordsVector.shape)
print(len(wordsList))


# ## Load Data
# 
# Now we will load the movie review data. 
# The data comes from https://www.kaggle.com/c/word2vec-nlp-tutorial/data

# In[4]:


data = pd.read_table("movie_review_dataset/labeledTrainData/labeledTrainData.tsv", sep='\t')


# Split training and testing datasets
# 
# 
# Set seed to make sure the split is always the same

# In[5]:


from sklearn.model_selection import train_test_split

train_reviews, test_reviews = train_test_split(data, test_size=0.4, train_size=0.6, random_state=3957)


# In[ ]:


# train_reviews.head()


# In[ ]:


# train_reviews.shape


# In[ ]:


# test_reviews.head()


# In[ ]:


# test_reviews.shape


# ## Exploratory analysis
# 
# Exploring the number of words in each review 

# In[6]:


def review_len(review):
    review = review.translate(string.punctuation)
    review = review.split(" ")
    return len(review)


# In[7]:


num_words = train_reviews.apply(lambda row: review_len(row['review']), axis=1)


# In[8]:


num_words.describe()


# In[9]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist(num_words, 50)
plt.xlabel("Sentence Length")
plt.ylabel("Frequency")
plt.axis([0, 1200, 0, 8000])
plt.show()


# In[10]:


maxSeqLength = 250
numDimensions = 300


# ### Converting sentences
# 
#  Creating a utility function to convert reviews into a numpy array of words.

# In[11]:


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
            sentenceList[index_count] = 399999 # Vector for unkown words
            
        index_count = index_count + 1
        
    return np.array(sentenceList)


# In[12]:


testSent = "Hello, how are you doing today?"

testSentVec = convert_sentence(testSent)

print(testSentVec)


# In[13]:


with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordsVector, testSentVec).eval().shape)


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

# In[ ]:


# subset_reviews = train_reviews.iloc[0:500]


# Using apply wasn't working correctly so I decided to use a list comprehension

# In[ ]:


# subset_reviews_ids = [convert_sentence(row[3]) for row in subset_reviews.itertuples()]


# In[ ]:


# subset_reviews_ids = pd.DataFrame(subset_reviews_ids)
# subset_reviews_ids.shape


# In[ ]:


# subset_reviews_ids.head(2)


# Now apply it to whole dataset. 
# 
# This takes a **LONG** time to run therefore, I have saved the output files as a csvs which can be loaded to avoid this step.

# #### Converting training data

# In[ ]:


# train_reviews_ids = [convert_sentence(row[3]) for row in train_reviews.itertuples()]
# train_reviews_ids_df = pd.DataFrame(train_reviews_ids)
# train_reviews_ids_df.to_csv("movie_review_dataset/labeledTrainData/train_ids_matrix.csv", index=False)


# #### Converting testing data

# In[ ]:


# test_reviews_ids = [convert_sentence(row[3]) for row in test_reviews.itertuples()]
# test_reviews_ids_df = pd.DataFrame(test_reviews_ids)
# test_reviews_ids_df.to_csv("movie_review_dataset/labeledTrainData/test_ids_matrix.csv", index=False)


# In[ ]:


# train_reviews_ids_df.shape


# In[ ]:


# train_reviews_ids_df.head()


# In[ ]:


# test_reviews_ids_df.shape


# In[ ]:


# test_reviews_ids_df.head()


# In[14]:


train_reviews_ids_df = pd.read_csv("movie_review_dataset/labeledTrainData/train_ids_matrix.csv")


# In[15]:


test_reviews_ids_df = pd.read_csv("movie_review_dataset/labeledTrainData/test_ids_matrix.csv")


# ## RNN Model

# In[16]:


def get_trainable_param(text):
    for tf_var in  tf.trainable_variables():
        if text in tf_var.name:
            return tf_var


def get_l2_regularizer():
    return sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if "kernel" in tf_var.name)
            


# Setting hyper parameters

# In[27]:


batch_size = 24
lstm_units = 64
num_classes = 2
itterations = 50000

# Dropout params
do_in = 0.7
do_out = 0.6
do_state = 1
lambda_l2 = 0.00015


# In[28]:


import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, num_classes])
input_data = tf.placeholder(tf.int32, [batch_size, maxSeqLength])


# In[29]:


data = tf.Variable(tf.zeros([batch_size, maxSeqLength, numDimensions]), dtype=tf.float32, name="data", trainable=False)
data = tf.nn.embedding_lookup(wordsVector, input_data)


# In[30]:


lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=do_out, input_keep_prob=do_in, state_keep_prob=do_state)
value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)


# In[31]:


weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]), name="kernel")
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bias")
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)


# In[32]:


correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


# In[33]:


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
#l2 = tf.nn.l2_loss(tf.trainable_variables()[0])
l2 = get_l2_regularizer()
loss = loss + (lambda_l2 * l2)
optimizer = tf.train.AdamOptimizer().minimize(loss)


# In[34]:


import datetime

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
train_logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_train_batchsize{}_lstmunits{}_doin{}_dostate{}_doout{}_regul{}".format(batch_size, lstm_units, do_in, do_state, do_out, lambda_l2)) + "/"
test_logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_test_batchsize{}_lstmunits{}_doin{}_dostate{}_doout{}_regul{}".format(batch_size, lstm_units, do_in, do_state, do_out, lambda_l2)) + "/"
train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
test_writer = tf.summary.FileWriter(test_logdir, sess.graph)


# ## Train Network
# 
# 

# In[35]:


from random import randint

# Helper functions to provide data for batch
def getTrainBatch():
    train_reviews.index = train_reviews_ids_df.index
    positive_review_ids = train_reviews_ids_df[train_reviews['sentiment'] == 1]
    negative_review_ids = train_reviews_ids_df[train_reviews['sentiment'] == 0]
    
    labels = []
    arr = np.zeros([batch_size, maxSeqLength])
    
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(0, positive_review_ids.shape[0]-1)
            arr[i] = positive_review_ids[num:num+1].as_matrix() 
            labels.append([1, 0])
        else:
            num = randint(1, negative_review_ids.shape[0]-1)
            arr[i] = negative_review_ids[num:num+1].as_matrix()
            labels.append([0, 1])
    
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batch_size, maxSeqLength])
    
    for i in range(batch_size):
        num = randint(0, test_reviews_ids_df.shape[0]-1)
        arr[i] = test_reviews_ids_df[num:num+1].as_matrix()
        # OMFG WE SHOULD BE INDEXING WITH num not i, here is why this thing wasn't working properlly.
        # TDO change this back to 1 liner
        if test_reviews.iloc[num, 1] == 1:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
    return arr, labels


# In[ ]:


sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(itterations):
    # Get next batch of reviews
    nextBatch, nextBatchLabels = getTrainBatch()
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
    
    #Write summary to Tensorboard
    if (i % 50 == 0):
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        train_writer.add_summary(summary, i)
    
    if (i % 500 == 0):
        testBatch, testBatchLabels = getTestBatch()
        summary_test = sess.run(merged, {input_data: testBatch, labels: testBatchLabels})
        test_writer.add_summary(summary_test, i)
        
    #Save the network every 10,000 training iterations
    if(i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step = 1)
        print("saved to %s" % save_path)

train_writer.close()
test_writer.close()


# In[ ]:




