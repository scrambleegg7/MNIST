#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

import os
import unittest
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt

from caffeBase.argumentClass import argumentClass

def readTFRecords():

    FILE = "TFRecords"
    filename_queue = tf.train.string_input_producer([ FILE ], num_epochs=None)

#    filename_queue = tf.train.string_input_producer([data_path])


    image, height,width, depth = read_and_decode(filename_queue)
    init = tf.global_variables_initializer()
    init2 = tf.initialize_local_variables()

    with tf.Session() as sess:
        sess.run([init,init2]  )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess)

        h,w,d = sess.run(  [height,width, depth]  )
        img = sess.run(image)

        print(h,w,d)
        print(len(img))
    """
    image = tf.reshape(image, tf.stack([height*3,width,3]) )
    image.set_shape([672,224,3])

    init = tf.global_variables_initializer()
    init2 = tf.initialize_local_variables()

    with tf.Session() as sess:
        sess.run([init,init2]  )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)

        for i in range(10):
            example = sess.run([image])

            img = Image.fromarray(example,"RGB")

        coord.request_stop()
        coord.join(threads)
    """

def read_and_decode(filename_queue,INPUT_IMAGE_SIZE=224):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
                serialized_example,
                features={
                    'height': tf.FixedLenFeature( [], tf.int64 ),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'depth': tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string),
#                    'label': tf.FixedLenFeature([], tf.int64),
                })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    # tf pack is discontinued function ....
    # rename to stack
    imshape = tf.stack( [ height, width, depth] )

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    #image = tf.reshape(image, imshape)
    #image.set_shape( [ 256, 256, 3] )

    #features = tf.parse_single_example(value, features={'image_raw': tf.FixedLenFeature([], tf.string)})
    #image = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)
    #image.set_shape([INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3])
    #IMAGE_HEIGHT=224
    #IMAGE_WIDTH=224
    #image = tf.image.resize_image_with_crop_or_pad(image=image,
    #                                   target_height=IMAGE_HEIGHT,
    #                                   target_width=IMAGE_WIDTH)

    # OPTIONAL: Could reshape into a 128x128 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    #image = tf.cast(image, tf.float32)
    #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    #label = tf.cast(features['label'], tf.int32)

    return image, height,width, depth



def main():
    readTFRecords()

if __name__ == "__main__":
    main()
