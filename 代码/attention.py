import tensorflow as tf
from tensorflow.python.data.util import nest
from tensorflow.python.keras.layers.core import Activation, Dense, Lambda, Permute, Reshape
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.layers.merge import Add, Concatenate, multiply
from tensorflow.keras import backend
from tensorflow.python.keras.layers.convolutional import Conv2D

# 通道注意力机制
def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    # channel = input_feature._keras_shape[channel_axis]
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio, # 神经元个数C/r, 激活函数reLU
                            kernel_initializer='he_normal',
                            activation = 'relu',
                            use_bias=True,
                            bias_initializer='zeros')

    shared_layer_two = Dense(channel, # 神经元个数C
                            kernel_initializer='he_normal',
                            use_bias=True,
                            bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel) # B*1*1*C
    avg_pool = shared_layer_one(avg_pool) # MLP
    assert avg_pool.shape[1:] == (1,1,channel//ratio) # B*1*1*C
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel) # B*1*1*C
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel) # B*1*1*C
    
    cbam_feature = Add()([avg_pool,max_pool]) # 相加
    cbam_feature = Activation('hard_sigmoid')(cbam_feature) # 激活函数 得到权重系数Mc
    
    if backend.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature]) # (B, W, H, C) * (B, 1, 1, C)

# 空间注意力机制
def spatial_attention(input_feature):
	kernel_size = 7
	if backend.image_data_format() == "channels_first":
		channel = input_feature.shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature.shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: backend.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool.shape[-1] == 1 # (B, W, H, 1)
	max_pool = Lambda(lambda x: backend.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool.shape[-1] == 1 # (B, W, H, 1)
	concat = Concatenate(axis=3)([avg_pool, max_pool]) 
	assert concat.shape[-1] == 2 # (B, W, H, 2)
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					activation = 'hard_sigmoid',
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False)(concat) 
	assert cbam_feature.shape[-1] == 1 # (B, W, H, 1) 权重系数 Ms
	
	if backend.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature]) # (B, W, H, C)

# 构建注意力模型CBAM: Convolutional Block Attention Module 
def cbam_block(cbam_feature,ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in CBAM: Convolutional Block Attention Module.
	"""
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature, )
	return cbam_feature