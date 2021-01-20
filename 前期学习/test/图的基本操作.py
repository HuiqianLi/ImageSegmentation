import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽这个警告，烦

import numpy as np
import tensorflow as tf
c = tf.constant(0.0)

g = tf.Graph() # 建立图
with g.as_default():
    c1 = tf.constant(0.0)
    print(c1.graph)
    print(g)
    print(c.graph)

g2 = tf.get_default_graph() # 获得图
print(g2)

tf.reset_default_graph() # 重置图
g3 = tf.get_default_graph()
print(g3)

print(c1.name) # 将c1的名字放到get_tensor_by_name里来反向得到其张量
t = g.get_tensor_by_name(name = "Const:0") # 获取张量
print(t)

# 获取节点
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

tensor1 = tf.matmul(a, b, name='exampleop') # 张量
print(tensor1.name, tensor1) # 先将张量及其名字打印出来
test = g3.get_tensor_by_name("exampleop:0")
print(test) # 此时test和tensor1是一样的

print(tensor1.op.name) # 获得OP的名字
testop = g3.get_operation_by_name("exampleop")
print(testop) # OP其实是描述张量中的运算关系，是通过访问张量的属性找到的

with tf.Session() as sess:
    test = sess.run(test)
    print(test)
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print(test)

    tt2 = g.get_operations() # 获取元素列表
    print(tt2)

    tt3 = g.as_graph_element(c1)
    print(tt3)