import tensorflow as tf
from mydataset2 import *
import os
from tensorflow.python.framework import graph_util
import numpy as np
dataset2 = MyDataset('test_data.csv', batch_size=1)
test_x, test_y = dataset2[0]

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], "./model_simple")
    graph = tf.get_default_graph()
    x = sess.graph.get_tensor_by_name('myInput:0')
    y = sess.graph.get_tensor_by_name('myOutput:0')
    out= sess.run(y, feed_dict={x: test_x})
