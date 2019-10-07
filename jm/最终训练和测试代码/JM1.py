
# 2019.9.22.9:37模型结构更改：去除所有激活函数采用非线性训练。
# 外部库

import numpy as np
from mydataset3 import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
# import matplotlib.pyplot as plt
# 设置所使用的GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,2"


# 超参数
learning_rate = 0.001
num_steps = 100000
# batch_size = 2000
display_step = 500
lam = 0.001
b1 = 0.74
b2 = 0.9
# 网络常数
num_input = 14 
num_classes = 1  
dropout = 0.5  # Dropout, probability to keep units
pb_file_path = './model'
# 输入变量
X = tf.placeholder(tf.float32, [None, num_input],name = 'myInput')
Y = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)

np.set_printoptions(threshold=np.inf)  
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
    l2 = tf.layers.dense(l1, 1024, tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    l3 = tf.layers.dense(l2, 512, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    l4 = tf.layers.dense(l3, 256, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    # l5 = tf.layers.dense(l4, 128, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    # l6 = tf.layers.dense(l5, 256, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    # l7 = tf.layers.dense(l6, 128, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    output = tf.layers.dense(l4, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    # 这里没有再调用softmax激活函数，因为代价函数自己会做softmax，又不用输出层就能得出分类情况，因此就没有再调用softmax
    return output


# 定义训练过程：

dataset1 = MyDataset('train_data_new_split.csv', batch_size=1000)
dataset2 = MyDataset('test_data_new_split.csv', batch_size=100000)
test_x, test_y = dataset2[0]
# print("test_x",test_x)

outputs = dnn3(X, lam)
tf.identity(outputs, name="myOutput")
loss0 = tf.losses.mean_squared_error(Y, outputs)  # compute cost
loss1 = tf.losses.get_regularization_loss()
loss = loss0 + loss1
# train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1= b1, beta2= b2)
gvs = opt.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
train = opt.apply_gradients(capped_gvs)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# 执行神经网络：
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

num_batch = len(dataset1)
f = open('data_loss.txt','w') 

with tf.Session(config=config) as sess:
    # 开始训练：
    sess.run(init_op)
    for step in range(1, num_steps+1):
        k = step % num_batch
        b_x, b_y = dataset1[int(k-1)]
        los, _, out = sess.run([loss, train, outputs], feed_dict={X: b_x, Y: b_y})
        if step % display_step == 0 or step == 1:
            test_loss = calc_mse(sess, test_x, test_y)
            print('Epoch:', step, '| train loss:', los, '| test loss:', test_loss)
            i = step // display_step
            f.write(str(i)+ ","+str(los)+","+ str(test_loss) + "\n")
            # print("input:\n", b_x[:10,:])
            # print("label:\n", b_y[:10])
            # print("outputs:\n", out[:10])
    print('******************************************************************************************')
     # 测试最终模型
    test_loss = calc_mse(sess, test_x, test_y)
    print('Epoch:', step, '| train loss:', los, '| test loss:', test_loss)
    f.write(str(i)+ ","+str(los)+","+ str(test_loss) + "\n")
    print("input:\n", b_x[:20,:])
    print("label:\n", b_y[:20])
    print("outputs:\n", out[:20]) 
    f.close() 

    test_out = sess.run(outputs, feed_dict={X: test_x, Y: test_y})
    test_out = test_out[:500]
    test_y = test_y[:500]
    file2 = open('test_out.txt','w') 
    file2.write(str(test_out) + "\n")
    file2.close()
    
    file3 = open('test_label.txt','w') 
    file3.write(str(test_y) + "\n")
    file3.close()
    # 保存pd模型
    # 简单方法：
    tf.saved_model.simple_save(sess,
                            "./model_simple",
                            inputs={"myInput": X},
                            outputs={"myOutput": outputs})
    print('******************************************************************************************')

 
print("learning_rate:",learning_rate,"num_steps:",num_steps, "lam:",lam, "beta1:", b1,"beta2:",b2)
