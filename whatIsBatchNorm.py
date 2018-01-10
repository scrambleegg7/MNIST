#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import unittest
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data


def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
#


class whatisBatchNorm(unittest.TestCase):

    def testBatch(self):

        print("load mnist test data ....")
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        print("image 28x28 = 784")
        x = tf.placeholder(tf.float32, [None, 784])
        #y = tf.matmul(x, W) + b

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])


        x_image = tf.reshape(x, [-1,28,28,1])
        epsilon = 1e-3
        #
        # conv1
        #
        #input, input_siz, in_ch, out_ch, patch_siz, activation='relu'
        #   x,  (28, 28),     1,     32,    (5, 5),      activation='none'
        with tf.variable_scope('conv_1'):
            wshape = [5, 5, 1, 32]
            w_cv = tf.Variable(tf.truncated_normal(wshape, stddev=0.1),trainable=True)
            b_cv = tf.Variable(tf.constant(0.1, shape=[32]),trainable=True)

            conv1 = tf.nn.conv2d(x_image, w_cv, strides=[1, 1, 1, 1], padding='SAME') + b_cv

            batch_mean2, batch_var2 = tf.nn.moments(conv1,[0])

            zeros_ = tf.constant(0.0, shape=[32])
            ones_ = tf.constant(1.0, shape=[32])
            scale2 = tf.Variable(ones_,name="gamma",trainable=True)
            beta2 = tf.Variable(zeros_,name="beta",trainable=True)
            conv1_bn = tf.nn.batch_normalization(conv1,batch_mean2,batch_var2,beta2,scale2,epsilon)

            #conv1_out = tf.nn.relu(conv1_bn)
            # skip batch process
            conv1_out = tf.nn.relu(conv1)

            pool1 = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('conv_2'):
            wshape = [5, 5, 32, 64]
            w_cv = tf.Variable(tf.truncated_normal(wshape, stddev=0.1),trainable=True)
            b_cv = tf.Variable(tf.constant(0.1, shape=[64]),trainable=True)

            conv2 = tf.nn.conv2d(pool1, w_cv, strides=[1, 1, 1, 1], padding='SAME') + b_cv

            batch_mean2, batch_var2 = tf.nn.moments(conv2,[0])

            zeros_ = tf.constant(0.0, shape=[64])
            ones_ = tf.constant(1.0, shape=[64])
            scale2 = tf.Variable(ones_,name="gamma",trainable=True)
            beta2 = tf.Variable(zeros_,name="beta",trainable=True)
            conv2_bn = tf.nn.batch_normalization(conv2,batch_mean2,batch_var2,beta2,scale2,epsilon)

            #conv2_out = tf.nn.relu(conv2_bn)
            # skip batch process
            conv2_out = tf.nn.relu(conv2)

            pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')
            pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

        wshape = [7 * 7 * 64, 1024]
        w_cv = tf.Variable(tf.truncated_normal(wshape, mean=0.0, stddev=0.05),trainable=True)
        b_cv = tf.Variable(tf.constant(0.1, shape=[1024]),trainable=True)

        h_fc1 = tf.nn.relu(tf.matmul(pool2_flat, w_cv) + b_cv)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        wshape = [1024, 10]
        w_cv = tf.Variable(tf.truncated_normal(wshape, mean=0.0, stddev=0.05),trainable=True)
        b_cv = tf.Variable(tf.constant(0.1, shape=[10]),trainable=True)

        linarg = tf.matmul(h_fc1_drop, w_cv) + b_cv
        y_pred = tf.nn.softmax(linarg)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_pred),
                                            reduction_indices=[1]))
        loss = cross_entropy

        optimizer = tf.train.AdamOptimizer(1e-4)
            # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            sess.run(init)
            batch = mnist.train.next_batch(1)

            res = sess.run(linarg,feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0} )
            res_1 = sess.run(y_pred,feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0} )

            #res = res.ravel()
            #plt.hist(res,bins=100)
            #plt.show()
            print(res,res_1)

            for i in range(500):
                break
                batch = mnist.train.next_batch(50)
                if i%100 == 0:

                    train_accuracy = accuracy.eval(feed_dict={
                            x:batch[0], y_: batch[1], keep_prob: 1.0})

                    print("step %d, training accuracy %g"%(i, train_accuracy))

                train_op.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


if __name__ == "__main__":
    unittest.main()
