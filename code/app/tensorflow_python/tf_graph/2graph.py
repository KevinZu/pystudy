# -*- coding: utf-8 -*-  
import tensorflow as tf
 
g = tf.Graph()
with g.as_default():
    c = tf.constant(4.0)
 
sess = tf.Session(graph=g)
c_out = sess.run(c)
print(c_out)
print(g)
print(tf.get_default_graph())
print(g.as_default())

