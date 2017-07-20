
# coding: utf-8

# In[ ]:

"""

개념 : 
- RNN(Recurrent Neural Network)은 구조체, LSTM은 그 구조체에서 셀에 게이트 개념(입력, 과거값, 출력에 각각 가중치를 두고 학습하는 것)을 더하여 vanishing gradient를 해결.

목표 :
LSTM(many-to-one)을 이용하여 MNIST classification

관련 포스팅 : 
"""


# In[1]:

import tensorflow as tf
import numpy as np
import time


# In[2]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)


# In[3]:

# random seed값을 정해놓고 돌릴 때 마다 같은 값으로 초기화되도록 한다.
tf.set_random_seed(777)


# In[4]:

# 입력되는 이미지 사이즈 28*28
input_size = 28   # input size(=input dimension)는 셀에 입력되는 리스트 길이
input_steps = 28  # input step(=sequence length)은 입력되는 리스트를 몇개의 time-step에 나누어 담을 것인가?  
hidden_size = 64  # hidden size(=output size)는 각 셀로부터 계산을 통해 나온 출력값의 크기 (ht)
n_classes = 10    # classification label 개수


# In[5]:

learning_rate= 0.01  # 학습 속도
training_epochs = 10 # 전체 데이터를 몇번 반복하여 학습 시킬 것인가
batch_size = 128     # 한번에 모든 data를 읽어오기에 너무 큼, 학습속도에 영향, 너무 크면 수렴하기 힘들다
display_step = 10


# In[6]:

# input과 output이 들어갈 placeholder를 만든다
X = tf.placeholder(tf.float32, [None, input_steps, input_size]) # (128, 28, 28)
Y = tf.placeholder(tf.float32, [None, n_classes])


# In[7]:

# LSTM으로 부터 나온 output을 class 중 하나로 변환하는 one layer를 정의
W = tf.Variable(tf.random_normal([hidden_size, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))


# In[8]:

# lstm cell을 만들고
# forget_bias=1.0(default)이면 모두 잊어라
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)


# In[9]:

# 밑의 에러를 해결하려면 unstack 해야함, axis = 0 과 1의 차이는? 
# axis = 1이므로 input_step을 기준으로 (128, 28, 28)*1에서 (128, 28)*28로 풀어줌 
x = tf.unstack(X, input_steps, axis=1)

# rnn 구조를 통해 output을 받는다.
# TypeError: inputs must be a sequence => unstack을 하지 않으면 발생
# output의 shape : batch_size*hidden_size
outputs1, states1 = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)


# In[10]:

# logits, output을 size 10짜리 vector로 만들어 실제값과 비교할 수 있도록 한다.
pred = tf.matmul(outputs1[-1], W) + b


# In[ ]:

# 위의 과정 함수로 만들기

# def static_rnn(X, W, b):
#     X = tf.unstack(X, input_steps, axis=1)
#     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
#     outputs1, states1 = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
    
#     return tf.matmul(outputs1[-1], W) + b

# pred = static_rnn(X, W, b)


# In[11]:

# logits를 softmax한 예측값과 실제값의 차이를 cross_entropy(=cost/lossfunction)를 이용하여 차이를 비교 
# 왜 Y는 unstack 안해도 될까? 병렬처리가 필요 없기 때문
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))


# In[12]:

# AdamOptimizer를 이용하여 cost를 최소화
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[13]:

# Evaluation model

# 예측값과 실제값이 같으면 True, 다르면 False
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))

# 맞춘 확률!
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[14]:

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[15]:

global_step = 0

start_time = time.time()

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, input_steps, input_size))
        
        
        c = sess.run(cost, feed_dict={X:batch_x, Y:batch_y})
        _ = sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y}) 
        # TypeError: unhashable type: 'list' => 위의 unstack된 값이 들어가지 않았기 때문
    
        avg_cost += c/total_batch
        
        global_step += 1
    
    test_data = mnist.test.images.reshape((-1, input_steps, input_size))
    test_label = mnist.test.labels
    
    print('Eopch:{:2d}, cost={:9f}'.format((epoch+1), avg_cost))
    print('Accuracy:', accuracy.eval(session=sess, feed_dict={X:test_data, Y:test_label}))
    
end_time = time.time()
    
print("execution time :", (end_time - start_time))

