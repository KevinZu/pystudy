# -*- coding: utf-8 -*-  
import tensorflow as tf
from tensorflow.python.platform import gfile
 
#load graph
with gfile.FastGFile("./graph1.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with gfile.FastGFile("./graph2.pb",'rb') as f2:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f2.read())
    tf.import_graph_def(graph_def, name='')
 
sess = tf.Session()
c1_tensor = sess.graph.get_tensor_by_name("c1:0")
c1 = sess.run(c1_tensor)
c2_tensor = sess.graph.get_tensor_by_name("c2:0")
c2 = sess.run(c2_tensor)

c3 = sess.run(c1_tensor + c2_tensor)

print(c1)
print(c2)
print(c3)
