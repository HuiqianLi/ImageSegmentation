import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽这个警告，烦

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!') # 定义一个常量
sess = tf.Session() # 建立一个session
print(sess.run(hello))
sess.close()

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()  #保证sess.run()能够正常运行
# hello = tf.constant('hello,tensorflow')
# sess= tf.compat.v1.Session()            #版本2.0的函数
# print(sess.run(hello))

# with session的使用
a = tf.constant(3) # 定义常量3
b = tf.constant(4) # 定义常量4
with tf.Session() as sess: # 建立session
    print("相加： %i" % sess.run(a+b))
    print("相乘： %i" % sess.run(a*b))
    sess.close()

# 注入机制 feed
c = tf.placeholder(tf.int16)
d = tf.placeholder(tf.int16)
add = tf.add(c, d)
mul = tf.multiply(c, d)
with tf.Session() as sess:
    # 计算具体数值
    print("相加： %i" % sess.run(add,feed_dict={c:3, d:4}))
    print("相乘： %i" % sess.run(mul,feed_dict={c:3, d:4}))
    print(sess.run([mul, add], feed_dict={c:3, d:4}))

