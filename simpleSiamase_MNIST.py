#

import skimage
import skimage.io
import skimage.transform

import unittest
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops

import numpy as np
from facenet.src import facenet

from tensorflow.examples.tutorials.mnist import input_data # for data

from matplotlib import offsetbox
import matplotlib.pyplot as plt

import os

from utils import triplet_loss

class SiamaseTest(unittest.TestCase):



    def visualize(self,embed, x_test):

        # two ways of visualization: scale to fit [0,1] scale
        # feat = embed - np.min(embed, 0)
        # feat /= np.max(feat, 0)

        # two ways of visualization: leave with original scale
        feat = embed
        ax_min = np.min(embed,0)
        ax_max = np.max(embed,0)
        ax_dist_sq = np.sum((ax_max-ax_min)**2)

        print("min max dist square **2" , ax_min, ax_max, ax_dist_sq)

        plt.figure()
        ax = plt.subplot(111)
        shown_images = np.array([[1., 1.]])
        for i in range(feat.shape[0]):
            dist = np.sum((feat[i] - shown_images)**2, 1)
            if np.min(dist) < 3e-4*ax_dist_sq:   # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [feat[i]]]

            #print(shown_images)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(x_test[i], zoom=0.6, cmap=plt.cm.gray_r),
                xy=feat[i], frameon=False
            )
            ax.add_artist(imagebox)

        plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
        # plt.xticks([]), plt.yticks([])
        plt.title('Embedding from the last layer of the network')
        plt.show()



    def network(self, x):
        weights = []
        fc1 = self.fc_layer(x, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 2, "fc3")
        return fc3

    def fc_layer(self, bottom, n_weight, name):

        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)

        return fc

    def _loss(self):

        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss


    def testModels(self):
        batch_size = 32
        mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

        print("MNIST data siamase test.....")

        #print(batch_x1.shape,batch_y1.shape)
        #print(batch_y)

        self.x1 = tf.placeholder(tf.float32, [None, 784])
        self.x2 = tf.placeholder(tf.float32, [None, 784])
        self.x3 = tf.placeholder(tf.float32, [None, 784])
        self.y_ = tf.placeholder(tf.float32, [None])

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)
            scope.reuse_variables()
            self.o3 = self.network(self.x3)

        triplet_loss_ = triplet_loss(self.o1,self.o2,self.o3)
        #train_step = tf.train.GradientDescentOptimizer(0.01).minimize( self._loss() )
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize( triplet_loss_ )

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)


            for step in range(3000):
                batch_x1, batch_y1 = mnist.train.next_batch(batch_size)
                batch_x2, batch_y2 = mnist.train.next_batch(batch_size)
                batch_x3, batch_y3 = mnist.train.next_batch(batch_size)
                batch_yt = (batch_y1 == batch_y2).astype('float')
                batch_yt2 = (batch_y2 == batch_y3).astype('float')
                batch_yt3 = (batch_y1 == batch_y3).astype('float')

                anchors = []
                positives = []
                negatives = []
                for t_index, t in enumerate(batch_yt):
                    if batch_yt[t_index] and batch_yt2[t_index] and batch_yt3[t_index]:
                        continue
                    if t:
                        anchors.append( batch_x1[t_index] )
                        positives.append( batch_x2[t_index] )
                        negatives.append( batch_x3[t_index] )
                    if batch_yt2[t_index]:
                        anchors.append( batch_x2[t_index] )
                        positives.append( batch_x3[t_index] )
                        negatives.append( batch_x1[t_index] )
                    if batch_yt3[t_index]:
                        anchors.append( batch_x1[t_index] )
                        positives.append( batch_x3[t_index] )
                        negatives.append( batch_x2[t_index] )


                #print("data shape for triplet_loss:",
                #    np.array([anchors]).shape,np.array([positives]).shape,np.array([negatives]).shape)

                if step == 0:
                    x1 = self.o1.eval( feed_dict={ self.x1 : mnist.train.images[:batch_size] }  )

                #_, loss = sess.run( [train_step, self._loss()],
                #        feed_dict={ self.x1 : batch_x1, self.x2 : batch_x2, self.y_ : batch_yt }  )

                tr_loss = sess.run( triplet_loss_,
                        feed_dict={ self.x1 : anchors, self.x2 : positives, self.x3 : negatives }  )


                if step % 100 == 0:
                #    print("step %d loss : %.5f " %    (step,loss)   )
                    print("step %d , triplet_loss : %.5f " %    (step, tr_loss)   )

            x1_final = self.o1.eval( feed_dict={ self.x1 : mnist.train.images[:batch_size] }  )


            test_data = mnist.train.images[:batch_size].reshape( [-1,28,28] )
            print("test_data shape:",test_data.shape)
            self.visualize(x1,test_data)

            self.visualize(x1_final,test_data)

if __name__ == "__main__":
    unittest.main()
