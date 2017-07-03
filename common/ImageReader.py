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

import numpy as np
import random 

def read_image(filename, label, size=30):
    """读取单张图片
    """
    axis('off')
    data = Image.open(filename)
    data = data.resize((size, size))
    data = np.array(data)
    return data, label


class ImageBatchReader(object):
    """图像批量读取类
    """
    def __init__(self, filename_list, label_list, batch_size=300, process=5, pic_size=30):
        """
        Args:
            sess: 
            filename_list: list, 
            label_list: list, label_list中每个元素类型为int
            process: 多进程读取图像的进程数
        """
        if len(filename_list) != len(label_list):
            raise ValueError("filename_list和label_list长度不一致!!")
        
        self.img_num = len(filename_list)
        print "读取的文件列表长度为{}".format(self.img_num)
        self.label_num = len(set(label_list))
        print "读取的文件总共包含的label数为{}".format(self.label_num)
        
        self.filename_label_list = [(filename_list[i], label_list[i]) for i in xrange(self.img_num)]
        
        self.batch_size = batch_size
        self.process = process
        self.pic_size = pic_size
         
    def _multiReader(self, filename_label_list):
        """ 
            Args:
                filename_label_list:
            Returns:
                images: np.array, 一个batch的图像数据
                labels: np.array, 一个batch图像对应的label数据
        """
        pool = multiprocessing.Pool(self.process)
        one_batch = []
        for (filename, label) in filename_label_list:
            res = pool.apply_async(read_image, (filename, label, self.pic_size)).get()
            one_batch.append(res)
        pool.close()
        pool.join()
        images = np.array([item[0] for item in one_batch])
        labels = np.array([item[1] for item in one_batch])
        return images, labels

    def getRandomOneBatch(batch_size=None):
        """
        """
        if batch_size is None:
            batch_size = self.batch_size
        filename_label_list = random.sample(self.filename_label_list, batch_size) 
        return self._multiReader(filename_label_list)

    def getOneBatch(self):
        """ Get one data batch.
            Args:
            Returns:
                images: np.array, 一个batch的图像数据
                labels: np.array, 一个batch图像对应的label数据
        """
        filename_label_list = self.filename_label_list[:self.batch_size]
        self.filename_label_list = self.filename_label_list[:self.batch_size]
        self.filename_label_list += filename_label_list
        return self._multiReader(filename_label_list)
    
    def getAllData(self):
        return self._multiReader(self.filename_label_list)
     
if __name__ == "__main__":
    data, label = read_image("/export/sdb/shelldream/MNIST/mnist_data/IMG/imgs_test/0_7.png", 7)
    print data.shape 
    import matplotlib.pyplot as plt # plt 用于显示图片
    plt.imshow(data)
    plt.savefig("raw_img")
