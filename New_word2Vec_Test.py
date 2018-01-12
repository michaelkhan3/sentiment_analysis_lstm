
# coding: utf-8

# In[ ]:


import pandas as pd
import csv
import numpy as np
import string
import re
import tensorflow as tf


# In[ ]:


wordsVector = np.load('glove_word2Vec/wordVectors.npy')
wordsList = np.load('glove_word2Vec/wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]


# In[ ]:


print(wordsVector.shape)
print(len(wordsList))


# ## Load Data
# 
# Now we will load the movie review data. 
# The data comes from https://www.kaggle.com/c/word2vec-nlp-tutorial/data

# In[ ]:


data = pd.read_table("movie_review_dataset/labeledTrainData/labeledTrainData.tsv", sep='\t')


# Split dataset into training, development and testing
# 
# 
# Set seed to make sure the split is always the same.

# In[ ]:


from sklearn.model_selection import train_test_split

train_reviews, test_reviews = train_test_split(data, test_size=0.4, train_size=0.6, random_state=3957)

# split test set into dev and test
dev_reviews = test_reviews.iloc[(int(len(test_reviews)/2)):, :]
test_reviews = test_reviews.iloc[:(int(len(test_reviews)/2)), :]


# In[ ]:


# test_reviews.to_csv("movie_review_dataset/labeledTrainData/test_data.csv", index=False)


# In[ ]:


# train_reviews.head()


# In[ ]:


# train_reviews.shape


# In[ ]:


# test_reviews.head()


# In[ ]:


# test_reviews.shape


# In[ ]:


# dev_reviews.shape


# ## Exploratory analysis
# 
# Exploring the number of words in each review 

# In[ ]:


def review_len(review):
    review = review.translate(string.punctuation)
    review = review.split(" ")
    return len(review)


# In[ ]:


num_words = train_reviews.apply(lambda row: review_len(row['review']), axis=1)


# In[ ]:


num_words.describe()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist(num_words, 50)
plt.xlabel("Review Length")
plt.ylabel("Frequency")
plt.axis([0, 1200, 0, 5000])
plt.show()


# In[ ]:


maxSeqLength = 250
###### Try to change this to 50 and see what happens
numDimensions = 50


# ### Converting sentences
# 
#  Creating a utility function to convert reviews into a numpy array of words.

# In[ ]:


def convert_sentence(sentence):
    # Got from analysis above.
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


# In[ ]:


testSent = "Hello, how are you doing today?"

testSentVec = convert_sentence(testSent)

print(testSentVec)


# In[ ]:


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


# #### Converting development data

# In[ ]:


# dev_reviews_ids = [convert_sentence(row[3]) for row in dev_reviews.itertuples()]
# dev_reviews_ids_df = pd.DataFrame(dev_reviews_ids)
# dev_reviews_ids_df.to_csv("movie_review_dataset/labeledTrainData/dev_ids_matrix.csv", index=False)


# In[ ]:


# train_reviews_ids_df.shape


# In[ ]:


# train_reviews_ids_df.head()


# In[ ]:


# test_reviews_ids_df.shape


# In[ ]:


# test_reviews_ids_df.head()


# In[ ]:


train_reviews_ids_df = pd.read_csv("movie_review_dataset/labeledTrainData/train_ids_matrix.csv")
test_reviews_ids_df = pd.read_csv("movie_review_dataset/labeledTrainData/test_ids_matrix.csv")
dev_reviews_ids_df = pd.read_csv("movie_review_dataset/labeledTrainData/dev_ids_matrix.csv")


# In[ ]:





# ## RNN Model

# In[ ]:


def get_trainable_param(text):
    for tf_var in  tf.trainable_variables():
        if text in tf_var.name:
            return tf_var


def get_l2_regularizer():
    return sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if "kernel" in tf_var.name)
            


# Setting hyper parameters

# In[ ]:


batch_size = 24
lstm_units = 64
num_classes = 2
# 58K epochs looks to be optimal for this network
iterations = 58000
learning_rate = 0.001

# Dropout params
do_in = 0.7
do_out = 0.6
do_state = 1
lambda_l2 = 0.00015


# In[ ]:


import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, num_classes])
input_data = tf.placeholder(tf.int32, [batch_size, maxSeqLength])


# In[ ]:


data = tf.Variable(tf.zeros([batch_size, maxSeqLength, numDimensions]), dtype=tf.float32, name="data", trainable=False)
data = tf.nn.embedding_lookup(wordsVector, input_data)


# In[ ]:


lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=do_out, input_keep_prob=do_in, state_keep_prob=do_state)
value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)


# In[ ]:


weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]), name="kernel")
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bias")
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)


# In[ ]:


correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


# In[ ]:


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
#l2 = tf.nn.l2_loss(tf.trainable_variables()[0])
l2 = get_l2_regularizer()
loss = loss + (lambda_l2 * l2)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# In[ ]:


import datetime

sess = tf.InteractiveSession()

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
train_logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_train_batchsize{}_lstmunits{}_doin{}_dostate{}_doout{}_regul{}_learnrate{}".format(batch_size, lstm_units, do_in, do_state, do_out, lambda_l2, learning_rate)) + "/"
dev_logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_dev_batchsize{}_lstmunits{}_doin{}_dostate{}_doout{}_regul{}_learnrate{}".format(batch_size, lstm_units, do_in, do_state, do_out, lambda_l2, learning_rate)) + "/"
train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
dev_writer = tf.summary.FileWriter(dev_logdir, sess.graph)


# ## Train Network
# 
# 

# In[ ]:


from random import randint

# Helper functions to provide data for batch
def get_train_batch():
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

def get_dev_batch():
    labels = []
    arr = np.zeros([batch_size, maxSeqLength])
    
    for i in range(batch_size):
        num = randint(0, dev_reviews_ids_df.shape[0]-1)
        arr[i] = dev_reviews_ids_df[num:num+1].as_matrix()
        if dev_reviews.iloc[num, 1] == 1:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
    return arr, labels

def get_test_data():
    labels = [[1, 0] if i == 1 else [0, 1] for i in dev_reviews.iloc[:, 1]]
    return test_reviews_ids_df, labels


# In[ ]:



saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
    # Get next batch of reviews
    nextBatch, nextBatchLabels = get_train_batch()
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
    
    #Write summary to Tensorboard
    if (i % 50 == 0):
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        train_writer.add_summary(summary, i)
    
    if (i % 500 == 0):
        devBatch, devBatchLabels = get_dev_batch()
        summary_dev = sess.run(merged, {input_data: devBatch, labels: devBatchLabels})
        dev_writer.add_summary(summary_dev, i)
        
    #Save the network every 10,000 training iterations
    if(i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step = 1)
        print("saved to %s" % save_path)

train_writer.close()
dev_writer.close()


# In[ ]:




