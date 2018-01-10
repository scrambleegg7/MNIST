#
#import tensorflow as tf
#import numpy as np

#from tensorflow.examples.tutorials.mnist import input_data

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import mnist

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import tensorflow as tf
import matplotlib.pyplot as plt

from caffeBase.argumentClass import argumentClass
from caffeBase.envParam import envParamFlower
#from caffeBase.imageProcessClass import imageProcessClass

#import cv2
from PIL import Image

FLAGS = None

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def convTraining(_):

    mnist = loadData()

    data_set = mnist.train
    print(data_set.num_examples)

    images = data_set.images
    print(images.shape[0],images.shape[1],images.shape[2],images.shape[3])

    print(images[0].shape)

    convert_to_records(images)

def convert_to_records(images,IMAGE_SIZE=224):

    filename = "TFRecords"

    print("writing tfrecords....",filename)
    writer = tf.python_io.TFRecordWriter(filename)

    #imagelistDicts = self.argCls.readImageList()

    for idx in range(0,100,3):


        vs1 = np.vstack( ( images[idx],images[idx+1] ) )
        vs2 = np.vstack( ( vs1,images[idx+2] ) )


        rows,cols,depth = vs2.shape


        image_raw = vs2.tostring()


        if idx % 500 == 0 and idx > 0:
            print("%d records processed .." % idx)
            print(rows,cols,depth)
            #print(image_raw)

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'image_raw': _bytes_feature(image_raw)}))
#            'label': _int64_feature(int( v )),

        writer.write(example.SerializeToString())

    writer.close()
    print("writing done....")
    print("%d records written on tfrecords .." % idx )





def loadData():

    #return input_data.read_data_sets("MNIST_data/", one_hot=True)

    return mnist.read_data_sets("MNIST_data/",
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=5000)


def main(_):

    #loadImageDataFile()

    convTraining(_)

if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--data_dir', type=str, default='MNIST_data',
#                      help='Directory for storing input data')
#    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main)
