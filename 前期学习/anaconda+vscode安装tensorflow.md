## anaconda+vscode下安装tensorflow

#### 1.anaconda、vscode任意版本

#### 2.tensorflow安装（版本1.9）

##### 2.1配置tensorflow环境

首先 新建一个tensorflow虚拟工作环境 ，（可能是避免放在base环境下，建个新的环境运行吧）有两种方法，

一种是 **Anaconda navigator** 下图形界面操作，先新建，具体见链接里的教程，注意python版本3.6（3.5会导致keras的版本出问题，但是keras的版本又和tensorflow对应（；´д｀）ゞ），建好后找到environment工作环境，点击tensorflow右边的小箭头，进入虚拟工作环境，然后pip show tensorflow检查安装；[参考链接🔗](https://blog.csdn.net/qq_41662115/article/details/86420983)

另一种是**Anaconda Prompt**终端，使用命令新建tensorflow环境，命令如下：

```python
conda create -n tensorflow python=3.6
```

这里tensorflow只是一个名字，也可以取别的。建好后，再输入：

```
activate tensorflow
```

##### 2.2tensorflow安装

首先可以把anaconda的安装源改成清华镜像，在打开**Anaconda navigator** 首页的灰色按钮“Channels”，点击添加“http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/”，把默认的“default”删掉。这步不改也没关系，就是安装的时候会比较慢。

后面都直接在终端操作的，前提是activate tensorflow进入tensorflow虚拟环境。

> 清华镜像网站https://pypi.tuna.tsinghua.edu.cn/simple/
>
> ↑可以在清华镜像找到需要的包的版本，复制下载链接；或者直接pip install并指定版本，感觉前者方便一点，后者我也不会操作。

**tensorflow安装：**

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/packages/e7/88/417f18ca7eed5ba9bebd51650d04a4af929f96c10a10fbb3302196f8d098/tensorflow-1.9.0-cp36-cp36m-win_amd64.whl#sha256=51aa006ce0c7cbca3381e05bc7658f59cfec90a11480f2d35afd342cef8294d8
```

这里的cp36好像是python版本，之前安的python3.5是用的cp35。

```
https://pypi.tuna.tsinghua.edu.cn/simple/tensorflow-gpu/
```

👆gpu

#### 3.kersa安装（版本2.2.0）

**keras安装：**

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/packages/68/12/4cabc5c01451eb3b413d19ea151f36e33026fc0efb932bf51bcaf54acbf5/Keras-2.2.0-py2.py3-none-any.whl#sha256=fa71a1f576dbd643532b872b8952afb65cc3ff7ed20d172e6b49657b710b43d0
```

#### 4.其他问题

运行测试代码：

测试代码：

使用vscode，先新建一个文件夹，右键用vscode打开，在里面新建一个py文件，这里注意，最下面蓝色的一条中有个环境的路径“Python3.6.2什么什么”，点击一下，在上面导航栏下方会出现几个选项，选择“tensorflow”那个，就是刚刚新建的tensorflow虚拟环境。

跑python代码的话，在扩展里搜一下关键字“python”，装一个python插件就可以了。

```python
# tensorflow2.0以上：
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  #保证sess.run()能够正常运行
hello = tf.constant('hello,tensorflow')
sess= tf.compat.v1.Session()            #版本2.0的函数
print(sess.run(hello))

# tensorflow2.0以下：
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# keras就直接import keras试试
```

跑这个代码的时候可能会遇到很多警告或者错误，多半是版本问题，百度搜一下就能解决。

之后的操作可以直接在vscode自带的终端进行，挺方便的。

#### 5.U-net示例代码

👇这两个对照着看可以跑出来的！注意文件结构

U-net：运行你的第一个U-net进行图像分割（keras实现）[🔗](https://blog.csdn.net/awyyauqpmy/article/details/79290710?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.compare&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.compare)

U-net入门--纠错过程[🔗](https://blog.csdn.net/weixin_45494335/article/details/103153244)