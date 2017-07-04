# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")
import os

from common import ImageReader
from autoEncoder import AutoEncoder
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def auto_encoder_train(filename_list, label_list, model_path, epoch_num=300):
    with tf.Session() as sess:
        auto_encoder = AutoEncoder.AdditiveGaussianNoiseAutoEncoder(sess=sess, n_input=img_dim, n_hidden=100)
        testReader = ImageReader.ImageBatchReader(filename_list, label_list, batch_size=500)

        for i in range(epoch_num): 
            images, labels = testReader.getOneBatch()
            shape_size = images.shape
            images = images.reshape(shape_size[0], shape_size[1]*shape_size[2]*shape_size[3])
            print "epoch: %i  loss: %f"%(i,auto_encoder.partial_fit(images))
        
        auto_encoder.save_model(model_path)

def auto_encoder_predict(filename_list, label_list, model_path, epoch_num=1):
    with tf.Session() as sess:
        auto_encoder = AutoEncoder.AdditiveGaussianNoiseAutoEncoder(sess=sess, n_input=img_dim, n_hidden=100)
        auto_encoder.load_model(model_path)

        testReader = ImageReader.ImageBatchReader(filename_list, label_list, batch_size=10, process=20)
        
        images, labels = testReader.getAllData()
        shape_size = images.shape
        images = images.reshape(shape_size[0], shape_size[1]*shape_size[2]*shape_size[3])
        #X = auto_encoder.getHidden(images)
        X = images
        y = labels
        
        sc = StandardScaler() # 估算每个特征的平均值和标准差
        sc.fit(X)
        X = sc.transform(X)
         
        offset = int(X.shape[0] * 0.9)  #90%
        X_train, y_train = X[:offset], y[:offset] #9成训练集
        X_test, y_test = X[offset:], y[offset:]   #1成测试集
        
        params = {
            'max_iter': 200,
            'penalty': 'l2',
            'solver' : 'lbfgs',
            'multi_class' : 'ovr'
        }
        classifier = LogisticRegression(**params)
        classifier.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, classifier.predict(X_test))
        print "accuracy: %f"%accuracy
         
 
if __name__ == "__main__":
    #test = "/export/sdb/shelldream/MNIST/mnist_data/IMG/imgs_test/"
    test = "/export/sdb/shelldream/MNIST/mnist_data/IMG/imgs_train/"
    model_path = "./model/mnist_auto_encoder.ckpt"
    #img_dim = 25600
    img_dim = 3600

    filename_list = []
    label_list = []

    for filename in os.listdir(test):
        label = int(filename.rstrip(".png").split("_")[1])
        filename_list.append(test + filename)
        label_list.append(label)
    
    #auto_encoder_train(filename_list, label_list, model_path)
    auto_encoder_predict(filename_list, label_list, model_path) 
