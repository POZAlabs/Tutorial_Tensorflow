
# coding: utf-8

# In[1]:

# 필요한 라이브러리를 로드합니다
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

np.random.seed(144)


# 기타변수를 초기화 합니다

learning_rate = 0.001
total_steps = 5000000
batch_size = 256 # 한 번에 받을 데이터 개수
# mnist 데이터는 28*28 
# 1b개의 28 vector 을 input length 로 설정하고 
# sequence 의 개수를 28개로 설정함
input_length = 28 
input_sequence = 28
n_hidden = 64
input_classes = 10


# placeholder 와 variable 을 선언합니다.
X = tf.placeholder(tf.float32,[None, input_sequence, input_length])
Y = tf.placeholder(tf.float32,[None, input_classes])
w_fw = tf.Variable(tf.random_normal([n_hidden, input_classes]))
w_bw = tf.Variable(tf.random_normal([n_hidden, input_classes]))
biases = tf.Variable(tf.random_normal([input_classes]))

keep_prob = tf.placeholder(tf.float32)

# lstm cell 2개를 생성합니다
lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden, state_is_tuple = True)
lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden, state_is_tuple = True)
lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)

outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, X, dtype = tf.float32)
outputs_fw = tf.transpose(outputs[0], [1,0,2])
outputs_bw = tf.transpose(outputs[1], [1,0,2])

pred = tf.matmul(outputs_fw[-1],w_fw) +tf.matmul(outputs_bw[-1],w_bw) + biases


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


total_batch = int(mnist.train.num_examples/batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    
    # training starts here!
    while step * batch_size < total_steps:
        step = step + 1
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        # transform data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, input_length, input_sequence))
        
        # execute optimization (backprop)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, keep_prob:0.9})
        
        if step % 10 == 0:
            # calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
            
            # calculate batch loss
            loss = sess.run(cost, feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
            print ('Iterations: ' + str(step*batch_size) + ', Minibatch Loss= ' +                   '{:.6f}'.format(loss) + ', Training Accuracy= ' +                   '{:.5f}'.format(acc))
        if step % 50 == 0:
            rand_train_index = np.random.randint(0, len(mnist.test.images), size=batch_size)
            test_data = mnist.test.images[rand_train_index,:].reshape((-1, input_length, input_sequence))
            test_label = mnist.test.labels[rand_train_index,:]

            print ('Test Accuracy:', sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob:1.0}))




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



