# -*- coding: utf-8 -*-  
import tensorflow as tf
 
g1 = tf.Graph()
with g1.as_default():
    # 需要加上名称，在读取pb文件的时候，是通过name和下标来取得对应的tensor的
    c1 = tf.constant(4.0, name='c1')
 
g2 = tf.Graph()
with g2.as_default():
    c2 = tf.constant(20.0,name='c2')
 
with tf.Session(graph=g1) as sess1:
    print(sess1.run(c1))
with tf.Session(graph=g2) as sess2:
    print(sess2.run(c2))
 
# g1的图定义，包含pb的path, pb文件名，是否是文本默认False
tf.train.write_graph(g1.as_graph_def(),'.','graph1.pb',False)
tf.train.write_graph(g2.as_graph_def(),'.','graph2.pb',False)
#tf.train.write_graph(c2.as_graph_def(),'.','graph-2.pb',True)

