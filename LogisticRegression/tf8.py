# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


#number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# create a layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# define the accuracy of testing data
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

# define the placeholder for input to network
#   shape = [None, 784]
xs = tf.placeholder(tf.float32, [None, 784]) #28x28
ys = tf.placeholder(tf.float32, [None, 10])

# define hidden layer and output layer
# l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# for classification
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# define loss
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),
#                      reduction_indices=[1]))
# 使用SOFTMAX函数去求得损失loss函数，通常使用cross_entropy方式
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])) #loss


# define training process
training_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# # initialize the variables
init = tf.global_variables_initializer()

# define the session
sess = tf.Session()
sess.run(init)


# observe the convergence of loss
for i in range(1000):
    #提取出一部分的样本100个样本
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(training_step, feed_dict={xs: batch_xs, ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
