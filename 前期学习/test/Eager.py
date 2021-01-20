import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽这个警告，烦

# 启用Eager 动态图
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
