# coding: utf-8

# 필요한 라이브러리를 로드합니다
import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

np.random.seed(144)

# 기타변수를 초기화 합니다

learning_rate = 0.001
training_epochs = 10 # 전체 데이터를 몇번 반복하여 학습 시킬 것인가
batch_size = 256 # 한 번에 받을 데이터 개수

# model 
# 입력되는 이미지 사이즈 28*28
input_size = 28   # input size(=input dimension)는 셀에 입력되는 리스트 길이
input_steps = 28  # input step(=sequence length)은 입력되는 리스트를 몇개의 time-step에 나누어 담을 것인가?  
n_hidden = 128
n_classes = 10    # classification label 개수


# placeholder 와 variable 을 선언합니다.
X = tf.placeholder(tf.float32,[None, input_steps, input_size])
Y = tf.placeholder(tf.float32,[None, n_classes])

W = tf.Variable(tf.random_normal([n_hidden * 2, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

keep_prob = tf.placeholder(tf.float32)

# lstm cell 2개를 생성합니다. 각 셀은 Dropout을 시켜줘 Overfitting을 방지합니다.
lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden, state_is_tuple = True)
lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden, state_is_tuple = True)
lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)

outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, X, dtype = tf.float32)

#나온 결과 값을 [batch_size, n_steps, n_hidden] -> [n_steps, batch_size, n_hidden]
outputs_fw = tf.transpose(outputs[0], [1,0,2])
outputs_bw = tf.transpose(outputs[1], [1,0,2])

#BLSTM은 나오는 결과 값을 합쳐 준다.
outputs_concat = tf.concat([outputs_fw[-1], outputs_bw[-1]], axis=1)

pred = tf.matmul(outputs_concat,W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Training & Prediction
sess = tf.Session()
sess.run(tf.global_variables_initializer())

global_step = 0

start_time = time.time()

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, input_steps, input_size)).astype(np.float32)
        
        c, _ = sess.run([cost, optimizer], feed_dict={X:batch_x, Y:batch_y, keep_prob:0.9})
    
        avg_cost += c/total_batch
        
        global_step += 1
    
    test_data = mnist.test.images.reshape((-1, input_steps, input_size))
    test_label = mnist.test.labels
    
    print('Eopch:{:2d}, cost={:9f}'.format((epoch+1), avg_cost))
    print('Accuracy:', accuracy.eval(session=sess, feed_dict={X:test_data, Y:test_label, keep_prob:1.0}))
    
end_time = time.time()
    
print("execution time :", (end_time - start_time))

