
# coding: utf-8

# # Testing pre-trained network

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf


# ## Load test data
# 
# Now that the trained network is loaded the test data to be fed into it will be loaded.
# 
# The original data is required for the sentiment labels and the ids matrix will be fed into the neural network.

# In[2]:


test_reviews = pd.read_csv("movie_review_dataset/labeledTrainData/test_data.csv")
test_reviews_ids_df = pd.read_csv("movie_review_dataset/labeledTrainData/test_ids_matrix.csv")


# ## Load word vectors

# In[3]:


wordsVector = np.load('glove_word2Vec/wordVectors.npy')
wordsList = np.load('glove_word2Vec/wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]


# ## Set up the Tensorflow graph

# In[4]:


maxSeqLength = 250
numDimensions = 50

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


# In[5]:


tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, num_classes])
input_data = tf.placeholder(tf.int32, [batch_size, maxSeqLength])

data = tf.Variable(tf.zeros([batch_size, maxSeqLength, numDimensions]), dtype=tf.float32, name="data", trainable=False)
data = tf.nn.embedding_lookup(wordsVector, input_data)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=do_out, input_keep_prob=do_in, state_keep_prob=do_state)
value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]), name="kernel")
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bias")
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


# ## Helper function to get test batch

# In[6]:


def get_test_batch(batch_num):
    start_from = batch_num * batch_size
    end_at = start_from + batch_size
    arr = np.zeros([batch_size, maxSeqLength])
    labels = []
    
    for i in range(batch_size):
        num = start_from + i
        arr[i] = test_reviews_ids_df[num:num+1].as_matrix()
        if test_reviews.iloc[num, 1] == 1:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        
    return arr, labels


# ## Load saved neural network

# In[7]:


sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))


# ## Run test data through network

# In[8]:


iterations =  int(np.floor(len(test_reviews)/batch_size))
correct_predictions = []

for i in range(iterations):
    # Get next sequential batch of test reviews
    next_batch, next_batch_labels = get_test_batch(i)
#     acc = sess.run(accuracy, {input_data: next_batch, labels: next_batch_labels})
#     print(acc)
    correct = sess.run(correctPred, {input_data: next_batch, labels: next_batch_labels})
    correct_predictions.extend(correct.tolist())


# In[9]:


sum(correct_predictions) / len(correct_predictions)

