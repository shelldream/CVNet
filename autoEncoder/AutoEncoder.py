# -*- coding:utf-8 -*-
"""
    Description: 实现Denoising AutoEncoder(去噪自编码器)
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")
sys.path.append("../common")
sys.path.append("./common")

from tools import xavier_init
import os
import numpy as np
import tensorflow as tf

class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, sess, n_input, n_hidden, transfer_function=tf.nn.softplus, \
        optimizer=tf.train.AdamOptimizer(), scale=1.0):
        """
            Args:
                n_input:输入节点数
                n_hidden:隐层节点数
                transfer_function:激活函数
                optimizer:优化方法
                scale:噪声系数
            Returns:
                None
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        self.weights = self._initialize_weights()
        
        #定义网络结构
        self.x = tf.placeholder(tf.float32, [None, self.n_input]) #输入数据
        self.X = self.x + scale*tf.random_normal((n_input, )) #加性随机噪声
        hidden_y = tf.add(tf.matmul(self.X, self.weights["w1"]) , self.weights["b1"])
        self.hidden = self.transfer(hidden_y)
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights["w2"]), self.weights["b2"])
        #定义损失函数及优化方法
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = sess
        self.sess.run(init)

        #模型保存
        self.saver = tf.train.Saver()
        self.default_save_path = "./model/model_default.ckpt"
        if not os.path.exists("./model"):
            os.popen("mkdir -p ./model")
    

    def _initialize_weights(self):
        """初始化网络中的参数"""
        all_weights = dict()
        all_weights["w1"] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights["b1"] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights["w2"] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights["b2"] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        """用一个batch数据进行训练并返回当前的损失
            Args:
                X: 一个batch的训练数据
            Returns:
                cost: 当前训练的损失
        """
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X, self.scale: self.training_scale})
        return cost
    
    def cal_total_cost(self, X):
        """计算总的cost
            Args:
                X: 一个batch的训练数据
            Returns:
                cost: 当前网络的cost
        """
        cost = self.sess.run(self.cost, feed_dict = {self.x:X, self.scale: self.training_scale})
        return cost

    def getHidden(self, X):
        """返回隐含层的输出结果
            Args:
                X: 一个batch的训练数据
            Returns:
                hidden: 网络隐含层的输出结果
        """
        hidden = self.sess.run(self.hidden, feed_dict = {self.x:X, self.scale: self.training_scale})
        return hidden

    def reconstruction(self, X):
        """返回重构结果
            Args:
                X: 一个batch的训练数据
            Returns:
                reconst_res: 重构结果
        """
        reconst_res = self.sess.run(self.reconstruction, feed_dict = {self.x:X, self.scale: self.training_scale})
        return reconst_res

    def getWeights(self):
        """返回隐含层的weight"""
        return self.sess.run(self.weights["w1"])

    def getBiases(self):
        """返回隐含层的偏置系数"""
        return self.sess.run(self.weights["b1"])
    
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
        if save_path is not None:
            self.save_path = save_path
        else:
            self.save_path = self.default_save_path

        if os.path.exists(self.save_path):
            self.saver.restore(self.save_path)
            print "The model %s has been loaded!"%self.save_path
        else:
            print "The model %s does not exsit!"%self.save_path
