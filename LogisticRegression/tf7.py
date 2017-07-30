# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# create a layer
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.histogram_summary(layer_name+'/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.histogram_summary(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.histogram_summary(layer_name + '/outputs', outputs)
        return outputs

# initial value
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define the placeholder for input to network
#   inputs为框架
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# define hidden layer and output layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# define loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),
                     reduction_indices=[1]))
    # 将LOSS变化的情况显示在Event中
    tf.scalar_summary('loss', loss)

# define training process
with tf.name_scope('train'):
    training_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# initialize the variables
init = tf.global_variables_initializer()

# define the session
sess = tf.Session()

# 把所有summary的变量，打包合并在一起，并放在log中
merged = tf.merge_all_summaries()

# 把整个框架loading 到一个文件中，才能从文件中loading到浏览器
writer = tf.train.SummaryWriter("logs/", sess.graph)
sess.run(init)

#plot the real data
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(x_data, y_data)
#
# # 需要连续plot的时候
# plt.ion()
# plt.show()


# observe the convergence of loss
for i in range(1000):
    sess.run(training_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        # to see the step improvement
        # print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        # try:
        #     # remove
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass

        result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(result, i)


        # lines = ax.plot(x_data, prediction_value, 'r-', lw=5)

        # stop for while
        # plt.pause(0.3)