#tensorflow_face_conv.py文件
#coding=utf-8
import os
import logging as log
import matplotlib.pyplot as plt

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import cv2
 
SIZE = 64
#用于得到传递进来的真实的训练样本
x_data = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
y_data = tf.placeholder(tf.float32, [None, None])
 
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)
 
#权重函数
def weightVariable(shape):
    #从服从指定正太分布的数值中取出指定大小的数组
    init = tf.random_normal(shape, stddev=0.01)
    #定义了变量后的初始化变量
    return tf.Variable(init)
 
#bais函数
def biasVariable(shape):
    #从服从指定正太分布的数值中取出指定大小的数组
    init = tf.random_normal(shape)
    #定义了变量后的初始化变量
    return tf.Variable(init)
 
#卷积函数
def conv2d(x, W):
    #以W为卷积核，步长为1没有填充进行卷积
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #返回卷积后的值
 
#池化层函数
def maxPool(x):
    #以2x2为池化曾核的大小，以步长为2进行下采样
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
# 减少过拟合函数，随机让某些权重不更新
def dropout(x, keep):
 
    return tf.nn.dropout(x, keep)
 
#卷积层构建函数
def cnnLayer(classnum):
    # 第一层
    W1 = weightVariable([3, 3, 3, 32]) # 卷积核大小(3,3),输入通道(3),输出通道/卷积核的个数(32)
    b1 = biasVariable([32])            # 设置权重
    conv1 = tf.nn.relu(conv2d(x_data, W1) + b1) # 进行卷积并对其进行非线性化处理
    pool1 = maxPool(conv1)                      # 进行池化曾操作
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5) # 32 * 32 * 32 多个输入channel 被filter内积掉了
 
    # 第二层
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5) # 64 * 16 * 16
 
    # 第三层
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5) # 64 * 8 * 8
 
    # 全连接层
    Wf = weightVariable([8*8*64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)
 
    # 输出层
    Wout = weightVariable([512, classnum])
    bout = weightVariable([classnum])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out
 
#训练函数的设置
def train(train_x, train_y, tfsavepath):
    ''' train'''
    log.debug('train')  
    out = cnnLayer(train_y.shape[1])  #进行卷及处理
    #softmax_cross_entropy_with_logits计算loss是代价值，也就是我们要最小化的值
    #对所有损失求平均
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_data))
    #最速下降法让交叉熵下降，步长为0.01
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    #查看目标判断是否准确
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_data, 1)), tf.float32))
    #训练模型的保存
    saver = tf.train.Saver()
    with tf.Session() as sess:  #开启一个会话
        sess.run(tf.global_variables_initializer()) #初始化模型的所有参数
        batch_size = 10
        num_batch = len(train_x) // 10  #返回其整数结果
        for n in range(10):
            r = np.random.permutation(len(train_x)) #返回一个新的数组
            train_x = train_x[r, :]
            train_y = train_y[r, :]
 
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                _, loss = sess.run([train_step, cross_entropy],\
                                   feed_dict={x_data:batch_x, y_data:batch_y,
                                              keep_prob_5:0.75, keep_prob_75:0.75})
 
                print(n*num_batch+i, loss)
 
        # 获取测试数据的准确率
        acc = accuracy.eval({x_data:train_x, y_data:train_y, keep_prob_5:1.0, keep_prob_75:1.0})
        print('after 10 times run: accuracy is ', acc)
        saver.save(sess, tfsavepath)
 
def validate(test_x, tfsavepath):
    ''' validate '''
    output = cnnLayer(2)
    #predict = tf.equal(tf.argmax(output, 1), tf.argmax(y_data, 1))
    predict = output
 
    saver = tf.train.Saver()
    with tf.Session() as sess:
        
        saver.restore(sess, tfsavepath)
        res = sess.run([predict, tf.argmax(output, 1)],
                       feed_dict={x_data: test_x,
                                  keep_prob_5:1.0, keep_prob_75: 1.0})
        return res
 
if __name__ == '__main__':
    pass
