import tensorflow as tf

n_input_nodes = 2
n_output_nodes = 1
x = tf.placeholder(tf.float32, (None, 2))
y = tf.placeholder(tf.float32, (None, 2))
W = tf.Variable(tf.random_normal((n_input_nodes, n_output_nodes)))
b = tf.Variable(tf.zeros(n_output_nodes))
z = tf.matmul(x, W) + b
out = tf.sigmoid(z)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(z,y))
