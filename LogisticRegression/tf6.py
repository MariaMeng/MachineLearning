# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

# initial value
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define the placeholder for input to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# define hidden layer and output layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

# define loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),
                     reduction_indices=[1]))

# define training process
training_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# initialize the variables
init = tf.global_variables_initializer()

# define the session
sess = tf.Session()
sess.run(init)

#plot the real data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)

# 需要连续plot的时候
plt.ion()
plt.show()


# observe the convergence of loss
for i in range(1000):
    sess.run(training_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        # to see the step improvement
        # print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        try:
            # remove
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs : x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)

        # stop for while
        plt.pause(0.3)