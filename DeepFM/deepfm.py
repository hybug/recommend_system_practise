# -*- coding: utf-8 -*-
# @Time    : 2019/9/4 18:08
# @Author  : hanyu
# @mail    ：hanyu01@mail.jj.cn


import numpy as np
import tensorflow as tf
import sys
from recommend_system_practise.DeepFM import data_parse

class Args():
    feature_sizes = 100
    field_size = 15
    embedding_size = 256
    deep_layers = [512, 256, 128]
    epoch = 3
    batch_size = 64
    learning_rate = 1.0
    le_reg_rate = 0.01
    checkpoint_dir = './'
    is_training = True

class Model():
    def __init__(self, args):
        self.feature_sizes = args.feature_sizes
        # 选择多少维特征进行高维的特征组合
        self.field_size = args.field_size
        self.embedding_size = args.embedding_size
        self.deep_layers = args.deep_layers
        self.l2_reg_rate = args.l2_reg_rate

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.deep_activation = tf.nn.relu
        self.weight = dict()
        self.checkpoint_dir = args.checkpoint_dir

        # build model
        self.build_model()

    def build_model(self):
        self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feature_index')
        self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feature_value')
        self.label = tf.placeholder(tf.float32, shape=[None, None], name='label')

        # 特征向量化, 原论文中的V矩阵，也就是更加特征生成其相应的embedding向量
        self.weight['feature_weight'] = tf.Variable(
            tf.random_normal([self.feature_sizes, self.embedding_size], 0.0, 0.01), name='feature_weight'
        )

        # 一次项中的w系数
        self.weight['feature_first'] = tf.Variable(
            tf.random_normal([self.feature_sizes, 1], 0.0, 1.0), name='feature_first'
        )

        num_layer = len(self.deep_layers)
        # Deep网络初始input：把向量化后的特征进行拼接后输入模型，形状是n个特征*embedding_size
        input_size = self.field_size * self.embedding_size
        stddev = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        self.weight['weight_0'] = tf.Variable(
            tf.truncated_normal(shape=[input_size, self.deep_layers[0]], mean=0, stddev=stddev), dtype=tf.float32
        )
        self.weight['bias_0'] = tf.Variable(
            tf.constant(0.0, shape=[1, self.deep_layers[0]])
        )

        # 生成deep network中的weight和bias
        if num_layer != 1:
            for i in range(1, num_layer):
                stddev = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                self.weight[f'weight_{i}'] = tf.Variable(
                    tf.truncated_normal(shape=[self.deep_layers[i - 1], self.deep_layers[i]], mean=0, stddev=stddev), dtype=tf.float32
                )
                self.weight[f'bias_{i}'] = tf.Variable(
                    tf.constant(0.0, [1, self.deep_layers[i]]), dtype=tf.float32
                )

        # 最后一层全连接层的size = deep部分output_size + 一次项的output_size + 二次项的output_size
        last_layer_size = self.deep_layers[-1] + self.field_size + self.embedding_size
        init_method = np.sqrt(np.sqrt(2.0 / (last_layer_size + 1)))
        # 生成最后一层的结果
        self.weight['last_layer'] = tf.Variable(
            np.random.normal(loc=0, scale=init_method, size=(last_layer_size, 1)), dtype=np.float32)
        self.weight['last_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        # embedding part
        # embedding层 embedding_index实际为根据feat_index对应的feature_weight对应的值
        self.embedding_index = tf.nn.embedding_lookup(self.weight['feature_weight'], self.feat_index) # Batch*F*K
        self.embedding_part = tf.multiply(self.embedding_index, tf.reshape(self.feat_value, [-1, self.field_size, 1]))
        # [Batch*F*1] * [Batch*F*K] = [Batch*F*K]
        print('embedding_part:', self.embedding_part)
        # embedding_part: Tensor("Mul:0", shape=(?, 15, 256), dtype=float32)

        # first part FM的第一组成部分
        self.embedding_first = tf.nn.embedding_lookup(self.weight['feature_first'], self.feat_index)
        self.first_order = 

