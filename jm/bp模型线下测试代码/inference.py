from feature.extract_feature2 import *
import pandas as pd
import tensorflow as tf
# from mydataset2 import *
import os
from tensorflow.python.framework import graph_util
import numpy as np

file = 'raw_train_set/train_108401.csv'
data = pd.read_csv(file)
label = data.iloc[:,-1].values
print("label:",label)
feat = extract_feature(data)
np.savetxt('feat.txt', feat, fmt='%.3f')
print(feat)
# np.set_printoptions(threshold='nan')

# dataset2 = MyDataset('test_data.csv', batch_size=1)
# test_x, test_y = dataset2[0]
file=open('data_for_test.txt','w') 
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], "./")
    graph = tf.get_default_graph()
    x = sess.graph.get_tensor_by_name('myInput:0')
    y = sess.graph.get_tensor_by_name('myOutput:0')
    out = sess.run(y,feed_dict={x: feat})
    file.write(str(i)+ ","+str(los)+","+ str(test_loss) + "\n")
print("out:",out)