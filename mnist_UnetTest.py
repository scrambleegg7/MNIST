#



from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import unittest
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from PIL import ImageOps



class loaddataClass(unittest.TestCase):



    def pixel_wise_softmax_2(self,output_map):
        exponential_map = tf.exp(output_map)
        sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
        tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
        return tf.div(exponential_map,tensor_sum_exp)

    def deconv(self, decv_name, bottom, prev_layer, out_ch, batch_size):

        with tf.variable_scope(decv_name):

            in_height = int( bottom.get_shape()[1] )
            in_width = int( bottom.get_shape()[2] )
            in_channels = int( bottom.get_shape()[3] )
            out_channels = out_ch

            output_shape = [batch_size, in_height * 2, in_width * 2, out_channels]
            wshape = [3, 3, out_channels, in_channels]
            #filter_ = tf.get_variable("filter", shape=wshape, dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
            filter_ = tf.get_variable("w_cv", shape=wshape, initializer=tf.contrib.layers.xavier_initializer())

            trans6= tf.nn.conv2d_transpose(bottom, filter_, output_shape, strides=[1, 2, 2, 1], padding="SAME")
            #up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)

            trans_ = tf.concat([trans6, prev_layer], axis=3)
            #print("concat trans6 and conv4_out shape:", trans6.get_shape())
        return trans_


    def maxPool(self,bottom):

        pool_ = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
        return pool_

    def setConv(self,name,wshape):
        pass

    def setLayer(self,name, x_image, out_ch, in_ch, activation="relu",pooling=True, kernel_size=3):

        with tf.variable_scope(name):
            #out_ch = 32
            wshape = [kernel_size, kernel_size, in_ch, out_ch]
            #w_cv = tf.Variable(tf.truncated_normal(wshape, stddev=0.1),trainable=True)
            w_cv = tf.get_variable("w_cv", shape=wshape, initializer=tf.contrib.layers.xavier_initializer())
            b_cv = tf.get_variable(name="b_cv", shape=[out_ch], initializer=tf.zeros_initializer())
            conv1 = tf.nn.conv2d(x_image, w_cv, strides=[1, 1, 1, 1], padding='SAME') + b_cv

            if activation=="relu":
                res_out = tf.nn.relu(conv1)
            if activation=="sigmoid":
                res_out = conv1

            if pooling:
                self.conv_layers.append(res_out)
                res_out = self.maxPool(res_out)

            return res_out

    def modelCheck(self,model,orig_data, data_img):

        print(model.shape)
        res = model.ravel()
        orig_data = orig_data.ravel()

        res_img = np.reshape(model,[28,28])
        #print(orig_data)

        revert_img = np.zeros( orig_data.shape[0] )
        for i in range(orig_data.shape[0]):
            if orig_data[i] > 0:
                revert_img[i] = -0.5
            else:
                revert_img[i] = 0.5
        revert_img = np.reshape(revert_img,[28,28])

        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(12,4))
        ax1.hist(res,bins=100)
        ax2.imshow(res_img,cmap=cm.gray_r)
        ax3.hist(orig_data,bins=100)
        ax4.imshow(data_img,cmap=cm.gray_r)
        ax5.imshow(revert_img,cmap=cm.gray_r)
        plt.show()

    def testBatch(self):

        self.conv_layers = []

        print("load mnist test data ....")
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        batch = mnist.train.next_batch(1)

        data = batch[0] - 0.5

        #data = data

        print(data.shape)
        batch_size , dim = data.shape

        row, col = int(np.sqrt(dim)), int(np.sqrt(dim))

        data_img = np.reshape(data,[row,col])

        x = tf.placeholder(tf.float32, [None, dim])
        x_image = tf.reshape(x,[-1,row,col,1])

        pool1 = self.setLayer("conv1",x_image,32,1)
        pool2 = self.setLayer("conv2",pool1,64,32)
        #pool3 = self.setLayer("conv3",pool2,128,64)
        #pool4 = self.setLayer("conv4",pool3,256,128)
        conv3 = self.setLayer("conv3",pool2,128,64,"relu",False)

        print("length of saved conv layers ...", len(self.conv_layers))
        prev_layer = self.conv_layers[-1]
        deconv3 = self.deconv("deconv3",conv3,prev_layer,64,batch_size)
        conv4 = self.setLayer("conv4",deconv3,64,128,"relu",False)

        prev_layer = self.conv_layers[-2]
        deconv4 = self.deconv("deconv4",conv4,prev_layer,32,batch_size)
        conv5 = self.setLayer("conv5",deconv4,32,64,"relu",False)
        output_map = self.setLayer("conv6",conv5,1,32,"sigmoid",False,1)

        prediction = self.pixel_wise_softmax_2(output_map)

        res = tf.sigmoid(output_map)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)
            self.modelCheck( sess.run( res, feed_dict = {x:data} ), data, data_img   )

            pred = sess.run(prediction,     feed_dict = {x:data} )
            print(pred)



if __name__ == "__main__":
    unittest.main()
