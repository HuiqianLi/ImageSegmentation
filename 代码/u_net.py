import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import gc
from numpy import mean

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽这个警告
os.environ['CUDA_VISIBLE_DEVICES']='0' # 0为GPU编号，可根据GPU的数量和使用情况自行设置

class U_Net():
    def __init__(self):
        # 设置图片基本参数
        self.height = 256
        self.width = 256
        self.channels = 1
        self.shape = (self.height, self.width, self.channels)

        # 优化器
        optimizer = Adam(0.002, 0.5) # Adam(自适应移动估计=RMSprop+momentum)
        # 参数说明：tip:该算法是面对反馈神经网络的良好选择
        # adam = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0,amsgrad=False)
        # lr：大或等于0的浮点数，学习率learning_rate; lr可以使用退化学习率进行设置
        # beta_1/beta_2：浮点数， 0<beta<1，通常很接近1
        """退化学习率 # 初始0.01，循环次数global_step，每100次衰减0.9
        lr = 0.002
        optimizer = Adam(lr, 0.5)
        global_step = tf.Variable(0, trainable=False)
        init_lr = 0.01
        lr = tf.train.exponential_decay(init_lr, global_step=global_step, decay_step=100, decay_rate=0.9)"""

        # u_net
        self.unet = self.build_unet()  # 创建网络变量
        self.unet.compile(loss='mse',  # 损失函数:均方误差（L2 loss）或者'mean_squared_error'
                          optimizer=optimizer, # 优化器
                          metrics=[self.metric_fun]) # 评价指标
        self.unet.summary()

    def build_unet(self, n_filters=16, dropout=0.1, batchnorm=True, padding='same'):
    # n_filters是卷积核数量，dropout是Dropout层的参数，batchnorm控制是否进行标准化，padding控制卷积前后图片尺寸是否变化

        # 定义一个多次使用的卷积块
        def conv2d_block(input_tensor, n_filters=16, kernel_size=3, batchnorm=True, padding='same'):
            # kernel_size：设置卷积核的大小
            # padding如果是’valid’，卷积后图片会变小一圈
            # 卷积层 -> 标准化层 -> 激活函数
            # the first layer
            x = Conv2D(n_filters, kernel_size, padding=padding)(input_tensor)
            if batchnorm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # the second layer
            x = Conv2D(n_filters, kernel_size, padding=padding)(x)
            if batchnorm:
                x = BatchNormalization()(x)
            X = Activation('relu')(x)
            return X

        # def 可以定义一个注意力块

        # 构建一个输入
        img = Input(shape=self.shape)

        # 收缩路径, 4次: 卷积模块 -> 下采样 -> dropout层
        # contracting path
        c1 = conv2d_block(img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout * 0.5)(p1) # 上面参数设置了dropout=0.1,每次训练时忽略0.1*0.5的神经元，减少过拟合

        c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm, padding=padding)

        # 扩展路径, 4次: 上采样 -> 特征合并 -> dropout层 -> 卷积模块
        # extending path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)

        # 构建一个输出
        output = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        return Model(img, output)

    # 根据Dice计算公式，定义如下评价函数
    def metric_fun(self, y_true, y_pred):
        fz = tf.reduce_sum(2 * y_true * tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
        fm = tf.reduce_sum(y_true + tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
        return fz / fm

    def load_data(self):
        x_source = []  # 定义一个空列表，用于保存数据集
        x_label = []
        for file in glob('./DICOM2_GTVnx/*'): # 获取文件夹名称
            for file_label in glob(file +'/Label/*'): # # 获取文件夹中文件
                img = np.array(Image.open(file_label), dtype='float32') / 255
                x_label.append(img[150:406, 120:376]) #512*512裁成256
            for file_source in glob(file +'/Source/*'):
                img = np.array(Image.open(file_source), dtype='float32') / 255
                x_source.append(img[150:406, 120:376])
        x_source = np.expand_dims(np.array(x_source), axis=3)  # 扩展维度，增加第4维
        x_label = np.expand_dims(np.array(x_label), axis=3)  # 变为网络需要的输入维度(num, 256, 256, 1)
        np.random.seed(116)  # 设置相同的随机种子，确保数据匹配
        np.random.shuffle(x_source)  # 对第一维度进行乱序
        np.random.seed(116)
        np.random.shuffle(x_label)
        # print(len(x_source))
        # 图片有3300张，按9:1进行分配 训练集和测试集
        return x_source[:2900, :, :], x_label[:2900, :, :], x_source[2900:, :, :], x_label[2900:, :, :]

    def train(self, epochs=101, batch_size=10):
        os.makedirs('./weights', exist_ok=True)
        # 获得数据
        x_source, x_label, y_source, y_label = self.load_data()

        # 加载已经训练的模型
        # self.unet.load_weights(r"./best_model.h5")

        # 设置训练的checkpoint
        callbacks = [EarlyStopping(patience=100, verbose=2),
                     ReduceLROnPlateau(factor=0.5, patience=20, min_lr=0.00005, verbose=2),
                     ModelCheckpoint('./weights/best_model.h5', verbose=2, save_best_only=True)]
                    # 其中EarlyStopping表示提前结束，如果验证集loss连续100次未下降，训练终止；verbose表示显示方法；
                    # ReduceLROnPlateau表示降低学习率，如果验证集loss连续20次未下降，学习率变为原来的1/10；
                    # ModelCheckpoint表示检查点(可实现断点续训)，只保存最优的模型(save_best_only保存在验证集上性能最好的模型
                    # 在每个epoch后，保存模型到filepath
                    # 文件路径/文件名，monitor需要监视的值，verbose信息展示 1展示(默认)0不展示2为每个epoch输出一行记录

        # 进行训练
        results = self.unet.fit(x_source, x_label, batch_size=batch_size, epochs=epochs, verbose=2,
                                callbacks=callbacks, validation_split=0.1, shuffle=True)
                                # x_source：输入数据。如果模型只有一个输入，那么x的类型是numpy array
                                #          如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
                                # x_label：标签，numpy array
                                # batch_size：32, 整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
                                # epochs：101, 整数，数据迭代轮数 训练终止时的epoch值，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch
                                # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
                                # callbacks：list，这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
                                # validation_split：0.1, 0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。
                                #                   验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。
                                # shuffle：一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。

        # 绘制损失曲线
        loss = results.history['loss']
        val_loss = results.history['val_loss']
        metric = results.history['metric_fun']
        val_metric = results.history['val_metric_fun']
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        x = np.linspace(0, len(loss), len(loss))  # 创建横坐标
        plt.subplot(121), plt.plot(x, loss, x, val_loss)
        plt.title("Loss curve"), plt.legend(['loss', 'val_loss'])
        plt.xlabel("Epochs"), plt.ylabel("loss")
        plt.subplot(122), plt.plot(x, metric, x, val_metric)
        plt.title("metric curve"), plt.legend(['metric', 'val_metric'])
        plt.xlabel("Epochs"), plt.ylabel("Dice")
        plt.show()  # 会弹出显示框，关闭之后继续运行
        fig.savefig('./evaluation/curve.png', bbox_inches='tight', pad_inches=0.1)  # 保存绘制曲线的图片
        plt.close()

    def test(self, batch_size=1):
        # 将输出结果写入unet_dice.txt文件
        file_handle=open('./evaluation/unet_dice.txt',mode='a+',encoding='utf-8')
        file_handle.write('dice of test data is:\n')
        os.makedirs('./evaluation/test_result', exist_ok=True)
        self.unet.load_weights(r"weights/best_model.h5")
        # 获得数据
        x_source, x_label, y_source, y_label = self.load_data()
        test_num = y_source.shape[0]
        index, step = 0, 0
        self.unet.evaluate(y_source, y_label)
        n = 0.0
        dice_list = []
        while index < test_num:
            print('schedule: %d/%d' % (index, test_num))
            step += 1  # 记录训练批数
            mask = self.unet.predict(y_source[index:index + batch_size]) > 0.1

            # 计算dice系数并写入文本
            fz = 2 * np.sum(mask.squeeze() * y_label[index].squeeze())
            fm = np.sum(mask.squeeze()) + np.sum(y_label[index].squeeze())
            dice = fz / fm
            dice_list.append(dice)
            file_handle.write(str(step) + ':%.2f\n' %dice)

            mask_true = y_label[index, :, :, 0]
            if (np.sum(mask) > 0) == (np.sum(mask_true) > 0):
                n += 1
            mask = Image.fromarray(np.uint8(mask[0, :, :, 0] * 255))
            mask.save('./evaluation/test_result/' + str(step) + '.png')
            mask_true = Image.fromarray(np.uint8(mask_true * 255))
            mask_true.save('./evaluation/test_result/' + str(step) + 'true.png')
            index += batch_size
            gc.collect()
        acc = n / test_num * 100
        print('the accuracy of test data is: %.2f%%' % acc)
        file_handle.write('the dice of test data is: %.2f\n' % mean(dice_list)) # 平均dice
        file_handle.write('the accuracy of test data is: %.2f%%' % acc)
        file_handle.close()

    def test1(self, batch_size=1):
        self.unet.load_weights(r"weights/best_model.h5")
        # 获得数据
        x_source, x_label, y_source, y_label = self.load_data()
        test_num = y_source.shape[0]
        for epoch in range(5):
            rand_index = []
            while len(rand_index) < 3:
                np.random.seed()
                temp = np.random.randint(0, test_num, 1)
                if np.sum(y_label[temp]) > 0:  # 确保产生有肿瘤的编号
                    rand_index.append(temp)
            rand_index = np.array(rand_index).squeeze()
            fig, ax = plt.subplots(3, 3, figsize=(18, 18))
            for i, index in enumerate(rand_index):
                mask = self.unet.predict(y_source[index:index + 1]) > 0.1
                ax[i][0].imshow(y_source[index].squeeze(), cmap='gray')
                ax[i][0].set_title('network input', fontsize=20)
                # 计算dice系数
                fz = 2 * np.sum(mask.squeeze() * y_label[index].squeeze())
                fm = np.sum(mask.squeeze()) + np.sum(y_label[index].squeeze())
                dice = fz / fm
                ax[i][1].imshow(mask.squeeze())
                ax[i][1].set_title('network output(%.4f)' % dice, fontsize=20)  # 设置title
                ax[i][2].imshow(y_label[index].squeeze())
                ax[i][2].set_title('mask label', fontsize=20)
            fig.savefig('./evaluation/show%d_%d_%d.png' % (rand_index[0], rand_index[1], rand_index[2]),
                        bbox_inches='tight', pad_inches=0.1)  # 保存绘制的图片
            print('finished epoch: %d' % epoch)
            plt.close()


if __name__ == '__main__':
    unet = U_Net()
    # unet.train()    # 开始训练网络
    # unet.test()     # 评价测试集并检测测试集肿瘤分割结果
    unet.test1()  # 随机显示
