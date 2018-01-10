from tensorflow.python.layers import layers
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# Create the model
def my_nn(images, drop_rate):
   net = tf.layers.conv2d(images, 32, [5,5], padding='same',
                                activation=tf.nn.relu, name='conv1')
   net = tf.layers.max_pooling2d(net, pool_size=[2,2], strides=[2,2],
                                name='pool1')
   net = tf.layers.conv2d(net, 64, [5,5], padding='same',
                                activation=tf.nn.relu, name='conv2')
   net = tf.layers.max_pooling2d(net, pool_size=[2,2], strides=[2,2],
                                name='pool2')
   net = tf.reshape(net, [-1, 7*7*64])
   net = tf.layers.dense(net, 1024, activation=tf.nn.relu, name='dense1')
   net = tf.layers.dropout(net, rate=drop_rate)
   net = tf.layers.dense(net, 10, activation=None, name='dense2')
   return net

def inference(x, y_, keep_prob):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    drop_rate = 1.0 - keep_prob
    y_pred = my_nn(x_image, drop_rate)

    loss = tf.losses.softmax_cross_entropy(y_, y_pred)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, accuracy, y_pred
