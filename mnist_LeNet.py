# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")
import os

import tensorflow as tf
import numpy as np

from LeNet import LeNet
from common import ImageReader

def get_file_list(dir_name):
    filename_list = []
    label_list = []

    for filename in os.listdir(dir_name):
        label = int(filename.rstrip(".png").split("_")[1])
        filename_list.append(dir_name + filename)
        label_list.append(label)
    
    return filename_list, label_list

def data_transfer(images, labels):
    images = np.reshape(images, (images.shape[0], 28, 28, 1))
    images = 1.0 - images/255.0  #灰度变化
    labels_one_hot = []
    for item in labels:
        tmp = [0 for i in range(10)]
        tmp[item] = 1
        labels_one_hot.append(tmp)
    labels_one_hot = np.array(labels_one_hot)
    return images, labels_one_hot


def train(dir_name, epoch_num, model_path, batch_size=5000):
    filename_list, label_list = get_file_list(dir_name)
    image_reader = ImageReader.ImageBatchReader(filename_list, label_list, batch_size=batch_size, pic_size=28, mode="L", process=20)
    
    with tf.Session() as sess:
        lenet = LeNet.LeNet(sess=sess)
        for epoch_i in xrange(epoch_num):
            images, labels = image_reader.getRandomOneBatch()
            images, labels_one_hot = data_transfer(images, labels)
            
            cost = lenet.partial_fit(images, labels_one_hot, 0.5)
            accuracy = lenet.evaluate(images, labels_one_hot, 0.5)
            print "epoch %d    cost:%f  accuracy:%f"%(epoch_i, cost, accuracy)

        lenet.save_model(model_path)
        

def evaluate(dir_name, model_path):
    filename_list, label_list = get_file_list(dir_name)
    image_reader = ImageReader.ImageBatchReader(filename_list, label_list, batch_size=500, pic_size=28, mode="L")
    with tf.Session() as sess:
        lenet = LeNet.LeNet(sess)
        lenet.load_model(model_path)
        for i in xrange(20):
            images, labels = image_reader.getRandomOneBatch()
            images, labels_one_hot = data_transfer(images, labels)
            accuracy = lenet.evaluate(images, labels_one_hot, 1.0)
            print "total accuracy: %f"%accuracy

if __name__ == "__main__":
    test_dir = "/export/sdb/shelldream/MNIST/mnist_data/IMG/imgs_test/"
    train_dir = "/export/sdb/shelldream/MNIST/mnist_data/IMG/imgs_train/"
    model_path = "./model/mnist_lenet.ckpt"
    train(train_dir, 100, model_path)
    #evaluate(test_dir, model_path) 
