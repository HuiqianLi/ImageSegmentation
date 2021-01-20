import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽这个警告，烦

import tensorflow as tf

# get_variable 和 Variable的区别
# Variable
var1 = tf.Variable(1.0, name='firstvar')
print("var1: ", var1.name)
var1 = tf.Variable(2.0, name='firstvar')
print("var1: ", var1.name)
var2 = tf.Variable(3.0)
print("var2: ", var2.name)
var2 = tf.Variable(4.0)
print("var2: ", var2.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1=", var1.eval())
    print("var2=", var2.eval())
    # sess.close()

# get_variable (name, shape ,init)
# 关于shape: shape参数的个数应为维度数，每一个参数的值代表该维度上的长度
get_var1 = tf.get_variable("firstvar", [1], initializer=tf.constant_initializer(3.0))
print("get_var1: ", get_var1.name)
get_var1 = tf.get_variable("firstvar1", [1], initializer=tf.constant_initializer(4.0))
print("get_var1: ", get_var1.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("get_var1=", get_var1.eval())

# 使用variable_scope将它们隔开可以创建两个同样名字的变量
with tf.variable_scope("test1",): # 定义一个作用域test1
    var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
    with tf.variable_scope("test2",): # 支持嵌套
        var2 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

print("var1:", var1.name)
print("var2:", var2.name)

with tf.variable_scope("test1", reuse=True):
    var3 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
    with tf.variable_scope("test2",):
        var4 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

print("var3:", var3.name)
print("var4:", var4.name)

tf.reset_default_graph() # 图（一个计算任务）里面的变量清空

