# -*- coding:utf-8 -*-
"""
    Description: 图像读取类,目前只支持PNG格式  https://shartoo.github.io/tensorflow-inputpipeline/
    Author: shelldream
    Date:2017-05-19
"""
import sys
reload(sys).setdefaultencoding("utf-8")
import multiprocessing

import matplotlib
matplotlib.use('Agg')
from PIL import Image
from pylab import *

import tensorflow as tf
import numpy as np

def read_image(filename, label):
    """读取单张图片
    """
    axis('off')
    data = Image.open(filename)
    data = data.resize((80, 80))
    data = np.array(data)
    return data, label


class ImageBatchReader(object):
    """图像批量读取类
    """
    def __init__(self, filename_list, label_list, batch_size=300):
        """
        Args:
            filename_list: list, 
            label_list: list, label_list中每个元素类型为int
        """
        if len(filename_list) != len(label_list):
            raise ValueError("filename_list和label_list长度不一致!!")

        self.img_num = len(filename_list)
        print "读取的文件列表长度为{}".format(self.img_num)
        self.label_num = len(set(label_list))
        print "读取的文件总共的label数为{}".format(self.label_num)

        self.file_tensor = tf.convert_to_tensor(filename_list, dtype=tf.string)
        self.labels_tensor = tf.convert_to_tensor(label_list, dtype=tf.int64)
        self.batch_size = batch_size

    def getOneBatch(self, process=5):
        """ Get one data batch.
            Args:
                process: 多进程读取图像的进程数
            Returns:
                images: np.array, 一个batch的图像数据
                labels: np.array, 一个batch图像对应的label数据
        """
        with tf.Session() as sess:
            self.input_queue = tf.train.slice_input_producer([self.file_tensor, self.labels_tensor], shuffle=True)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(tf.initialize_all_variables())
            filename_label_list = []
            for i in range(self.batch_size):
                one_instance = sess.run(self.input_queue)
                filename_label_list.append(one_instance)
            coord.request_stop()
            coord.join(threads)
            pool = multiprocessing.Pool(process)
            one_batch = []
            for (filename, label) in filename_label_list:
                res = pool.apply_async(read_image, (filename, label)).get()
                one_batch.append(res)
            pool.close()
            pool.join()
            images = np.array([item[0] for item in one_batch])
            labels = np.array([item[1] for item in one_batch])
            return images, labels

if __name__ == "__main__":
    data, label = read_image("/export/sdb/huangxiaojun/MNIST/mnist_data/IMG/imgs_test/0_7.png", 7)
    print data.shape 
    import matplotlib.pyplot as plt # plt 用于显示图片
    plt.imshow(data)
    plt.savefig("raw_img")
