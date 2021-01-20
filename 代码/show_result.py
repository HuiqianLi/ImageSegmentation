from glob import glob
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt

    
def load_result():
    unet_result = []  # 定义一个空列表，用于保存数据集
    att_unet_result = []
    label = []
    for file in glob('./evaluation/test_result/*'): # 获取文件夹名称
        if file[-8:]!='true.png':
            img = np.array(Image.open(file), dtype='float32') / 255
            unet_result.append(img)
    for file in glob('./evaluation/test_result_att/*'): # 获取文件夹名称
        if file[-8:]!='true.png':
            img = np.array(Image.open(file), dtype='float32') / 255
            att_unet_result.append(img)
            img = np.array(Image.open(file[:-4] + 'true.png'), dtype='float32') / 255
            label.append(img)
    return unet_result, att_unet_result, label

def show_result():
    # 获得数据
    unet_result, att_unet_result, label = load_result() # 导入unet训练结果
    test_num = len(unet_result)
    for epoch in range(5):
        rand_index = []
        while len(rand_index) < 3:
            np.random.seed()
            temp = np.random.randint(0, test_num, 1)
            rand_index.append(temp)
        rand_index = np.array(rand_index).squeeze()
        fig, ax = plt.subplots(3, 3, figsize=(18, 18))
        for i, index in enumerate(rand_index):
            # # 计算dice系数
            # fz = 2 * np.sum(mask.squeeze() * x_label[index].squeeze())
            # fm = np.sum(mask.squeeze()) + np.sum(x_label[index].squeeze())
            # dice = fz / fm
            ax[i][0].imshow(unet_result[index].squeeze())
            ax[i][0].set_title('Unet output', fontsize=20)  # 设置title Unet
            ax[i][1].imshow(att_unet_result[index].squeeze())
            ax[i][1].set_title('Att_Unet output', fontsize=20)  # 设置title att_unet
            ax[i][2].imshow(label[index].squeeze())
            ax[i][2].set_title('mask label', fontsize=20)
        fig.savefig('./evaluation/show_result/show%d_%d_%d.png' % (rand_index[0], rand_index[1], rand_index[2]),
                    bbox_inches='tight', pad_inches=0.1)  # 保存绘制的图片
        print('finished epoch: %d' % epoch)
        plt.close()


if __name__ == '__main__':
    show_result()  # 随机显示
