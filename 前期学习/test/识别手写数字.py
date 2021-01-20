import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽这个警告，烦

# 导入图片数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print('输入数据：', mnist.train.images)
# print('输入数据的shape：', mnist.train.images.shape)
import pylab
import tensorflow as tf

# im = mnist.train.images[1]
# im = im.reshape(-1, 28)
# pylab.imshow(im)
# # pylab.show()

# print('输入数据的shape:', mnist.test.images.shape) # 测试数据集
# print('输入数据的shape:', mnist.validation.images.shape) # 验证数据集

tf.reset_default_graph()
# 定义占位符
x = tf.placeholder(tf.float32, [None, 784]) # MNIST数据集的维度是28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 数字0~9, 共10个类别
W = tf.Variable(tf.random_normal([784, 10])) # weight
b = tf.Variable(tf.zeros([10])) #bias
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax分类
#########################正向传播结束#########################

# 损失函数  将生成的pred与样本标签y进行一次交叉熵的运算，然后取平均值
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 定义参数
learning_rate = 0.01
# 使用梯度下降优化器  找到能够使这个误差最小化的b和W的偏移量
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#########################反向传播结束#########################

# 训练模型
training_epochs = 25 # 把整个训练样本集迭代25次
batch_size = 100 # 在训练过程中一次取100条数据进行训练
display_step = 1 # 每训练一次就把具体的中间状态显示出来

saver = tf.train.Saver()
model_path = "./log/521model.ckpt"

'''
# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # Initializing OP 初始化

    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 循环所有数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行优化器
            _, c = sess.run([optimizer, cost], feed_dict={x:batch_xs, y:batch_ys})
            # 计算平均loss
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            
    print("Finished!")

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

    # 保存模型
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
'''

print("Starting 2nd session...")
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 恢复模型变量
    saver.restore(sess, model_path)
    # 测试model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict={x: batch_xs})
    print(outputval, predv, batch_ys)
    # 第一个数组是输出的预测结果。
    # 第二个大的数组比较大，是预测出来的真实输出值。
    # 第三个大的数组元素都是0和1，是标签值onehot编码表示的结果1和结果2。

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()