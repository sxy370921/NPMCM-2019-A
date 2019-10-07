
# 基于框架1的一个DNN实现
# 该DNN结构：
# 外部库

import numpy as np
from mydataset2 import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
# import matplotlib.pyplot as plt
# 设置所使用的GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,2"

np.set_printoptions(threshold=np.inf)  
# 超参数
learning_rate = 0.01
num_steps = 100000
batch_size = 5000
display_step = 1000
lam = 0.001

# 网络常数
num_input = 53  
num_classes = 1  
dropout = 0.5  # Dropout, probability to keep units
pb_file_path = './model'
# 输入变量
X = tf.placeholder(tf.float32, [None, num_input],name = 'myInput')
Y = tf.placeholder(tf.float32, [None, 1], name = "true")
keep_prob = tf.placeholder(tf.float32)


# 定义功能函数:

# 准确率函数
# def compute_accuracy(sess0, out, v_xs, v_ys):
#     correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
#     accuracy0 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess0.run(accuracy0, feed_dict={X: v_xs, Y: v_ys})
#     return result

def calc_mse(sess0, x, y):
    mse = sess0.run(loss0, feed_dict={X: x, Y: y})
    return mse    


# 定义网络结构：

# dnn3是有两个隐藏层的网络，并且加入了L2正则化
def dnn3(x, lam0):
    l1 = tf.layers.dense(x, 256, tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    l2 = tf.layers.dense(l1, 512, tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    l3 = tf.layers.dense(l2, 512, tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    l4 = tf.layers.dense(l3, 1024, tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    l5 = tf.layers.dense(l4, 512, tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    l6 = tf.layers.dense(l5, 256, tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    l7 = tf.layers.dense(l6, 128, tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    output = tf.layers.dense(l7, 1, tf.nn.sigmoid, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    # 这里没有再调用softmax激活函数，因为代价函数自己会做softmax，又不用输出层就能得出分类情况，因此就没有再调用softmax
    return output


# 定义训练过程：

dataset1 = MyDataset('train_data.csv', batch_size=1000)
dataset2 = MyDataset('test_data.csv', batch_size=1)
test_x, test_y = dataset2[0]

outputs = dnn3(X, lam)
tf.identity(outputs, name="myOutput")
loss0 = tf.losses.mean_squared_error(Y, outputs)  # compute cost
loss1 = tf.losses.get_regularization_loss()
loss = loss0 + loss1
tf.identity(loss0, name="coloss")
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# 执行神经网络：
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

num_batch = len(dataset1)

with tf.Session(config=config) as sess:
    # 开始训练：
    sess.run(init_op)
    for step in range(1, num_steps+1):
        k = step % num_batch
        b_x, b_y = dataset1[int(k-1)]
        los, _ = sess.run([loss, train], feed_dict={X: b_x, Y: b_y})
        if step % display_step == 0 or step == 1:
            test_loss = calc_mse(sess, test_x, test_y)
            print('Epoch:', step, '| train loss:', los, '| test loss:', test_loss)
    print('******************************************************************************************')
    # 保存pd模型
    # 简单的tf.saved_model.simple_save方法：
    tf.saved_model.simple_save(sess,
                            "./model_simple",
                            inputs={"myInput": X, "true":Y},
                            outputs={"myOutput": outputs, "coloss":loss0})
    print('******************************************************************************************')
    # 测试最终模型
    test_loss = calc_mse(sess, test_x, test_y)
    print('END:', step,  'test MSE:', test_loss)

print("learning_rate:",learning_rate,"num_steps:",num_steps, "batch_size:", batch_size, "lam:",lam)
