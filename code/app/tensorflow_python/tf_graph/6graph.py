# -*- coding: utf-8 -*-  
import tensorflow as tf
 
g1 = tf.Graph()
with g1.as_default():
    # 声明的变量有名称是一个好的习惯，方便以后使用
    c1 = tf.constant(4.0, name="c1")
 
g2 = tf.Graph()
with g2.as_default():
    c2 = tf.constant(20.0, name="c2")
 
with tf.Session(graph=g2) as sess1:
    # 通过名称和下标来得到相应的值
    c1_list = tf.import_graph_def(g1.as_graph_def(), return_elements = ["c1:0"], name = '')
    print(sess1.run(c1_list[0]+c2))

