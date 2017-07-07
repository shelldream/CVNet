# -*- coding:utf-8 -*-
"""
    Description:
        实现LeNet5
        输入的图片大小限制为28*28
    Author: shelldream
    Date: 2017-07-04
"""
import sys
reload(sys).setdefaultencoding("utf-8")
sys.path.append("../common")
sys.path.append("./common")

from tools import xavier_init
import os
import numpy as np
import tensorflow as tf

class LeNet(object):
    def __init__(self, sess, transfer_function=tf.nn.relu, optimizer=tf.train.AdamOptimizer()):
        """
            Args:
                transfer_function:激活函数
                optimizer:优化方法
            Returns:
                None
        """
        self.transfer = transfer_function
        self.weights = self._initialize_weights()
        self.sess = sess
        
        #输入数据定义
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1]) #输入数据, 大小28*28， 1 channel 
        self.y_ = tf.placeholder(tf.float32, [None, 10])  # 输入数据 one-hot编码, 10分类
        self.keep_prob = tf.placeholder(tf.float32) #dropout 概率

        #定义网络结构
        h_conv1 = self.transfer(self._conv2d(self.x, self.weights["w_conv1"]) + self.weights["b_conv1"])
        h_pool1 = self._max_pool_2x2(h_conv1)
        
        h_conv2 = self.transfer(self._conv2d(h_pool1, self.weights["w_conv2"]) + self.weights["b_conv2"])
        h_pool2 = self._max_pool_2x2(h_conv2)
        
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = self.transfer(tf.matmul(h_pool2_flat, self.weights["w_fc1"]) + self.weights["b_fc1"])
        
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        self.y = tf.nn.softmax(tf.matmul(h_fc1_drop, self.weights["w_fc2"]) + self.weights["b_fc2"])

        #定义损失函数及优化方法
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y_*tf.log(self.y), reduction_indices=[1]))  #交叉熵
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        #模型指标,预测准确率
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        #模型保存
        self.saver = tf.train.Saver()
        self.default_save_path = "./model/model_default.ckpt"
        if not os.path.exists("./model"):
            os.popen("mkdir -p ./model")
    
    def _initialize_weights(self):
        all_weights = dict()
        all_weights["w_conv1"] = self._init_weight_variable([5, 5, 1, 32])
        all_weights["b_conv1"] = self._init_bias_variable([32])
        all_weights["w_conv2"] = self._init_weight_variable([5, 5, 32, 64])
        all_weights["b_conv2"] = self._init_bias_variable([64])
        all_weights["w_fc1"] = self._init_weight_variable([7*7*64, 1024])
        all_weights["b_fc1"] = self._init_bias_variable([1024]) 
        all_weights["w_fc2"] = self._init_weight_variable([1024, 10])
        all_weights["b_fc2"] = self._init_bias_variable([10]) 
        return all_weights
         
    def _init_weight_variable(self, shape, stddev=0.1):
        """按照截断的正态分布随机初始化权值"""
        initial = tf.truncated_normal(shape=shape, stddev=stddev)
        return tf.Variable(initial)
    
    def _init_bias_variable(self, shape, const=0.1):
        """以常数值初始化偏置"""
        initial = tf.constant(const, shape=shape)
        return tf.Variable(initial)
    
    def _conv2d(self, x, W):
        """卷积操作"""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
    
    def partial_fit(self, X, Y, keep_prob):
        """用一个batch数据进行训练并返回当前的损失
            Args:
                X: 一个batch的训练数据
                Y:
                keep_prob:
            Returns:
                cost: 当前训练的损失
        """
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X, self.y_: Y, self.keep_prob: keep_prob})
        return cost
    
    def evaluate(self, X, Y, keep_prob):
        """计算总的cost和accuracy
            Args:
                X: 一个batch的训练数据
                Y:
            Returns:
                accuracy: 当前网络预测结果的准确性
        """
        accuracy = self.sess.run(self.accuracy, feed_dict={self.x:X, self.y_:Y, self.keep_prob:keep_prob}) 
        return accuracy
    
    def predict(self, X):
        """预测给定图片的预测结果"""
        label = self.sess.run(self.y, feed_dict = {self.x:X, self.keep_prob:1.0}) 
        return label

    def save_model(self, save_path=None):
        """保存模型至特定路径"""
        if save_path is not None:
            self.save_path = save_path
        else:
            self.save_path = self.default_save_path
        self.saver.save(self.sess, save_path)
        print "The model has been saved in the file: %s"%self.save_path

    def load_model(self, save_path=None):
        """载入已有模型"""
        self.saver.restore(self.sess, save_path)
        print "The model %s has been loaded!"%save_path
        for k, v in self.weights.items():
            self.sess.run(v) 
        
