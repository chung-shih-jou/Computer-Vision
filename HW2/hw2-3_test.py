# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:26:03 2018

@author: COSH
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:10:30 2018

@author: COSH
"""
from PIL import Image
import glob 
import numpy as np 
from scipy.signal import argrelextrema
from numpy import linalg
import scipy
import sklearn.manifold as man
from sklearn import neighbors as nei
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import np_utils
from keras.models import load_model
import pandas as pd
import csv


def load_image(flag,Nclass):
    
    image_list=[]
    number=0
    for i in range(Nclass):
        path_test = 'C:/Users/COSH/Downloads/研究所/上課內容/Computer Vision/作業二/hw2-3_data/valid/class_'+str(i)+'/*'
        for filename in glob.glob(path_test):
            im = Image.open(filename)
            width,height = im.size
            a = np.array(im)
            image_list.append(a)
            number+=1
    image_list = np.reshape(image_list,(number,width,height))
    print("TestImage:"+str(image_list.shape))
    return image_list,number,width,height

def plot_images_labels_prediction(images,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')

        ax.set_title("predict="+str(prediction[idx])
                     ,fontsize=10) 
        
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

if __name__ == '__main__':
    
    Nclass = 10
    total = 10000
    x_test,Test_n,width,height = load_image(0,Nclass)
    X_Test = np.expand_dims(x_test, axis=-1) / 255.
    
#    for i in range(Nclass):
#        arr = np.array([i]*(total//10))
#        if i==0:
#            y_test = arr
#        else:
#            y_test = np.r_[y_test,arr]
    label = []
    for i in range(total):
        arr = '000'+str(i)
        label.append(arr)        
        
#    Y_test = np_utils.to_categorical(np.uint8(y_test)) 
    
    model = load_model('CNN_model.h5')
    prediction=model.predict_classes(X_Test)
    plot_images_labels_prediction(x_test,prediction,idx=0)
#    pd.crosstab(y_test,prediction,
#            rownames=['label'],colnames=['predict'])
    np.savetxt('scores.csv', [p for p in zip(label,prediction)], delimiter=',', fmt='%s')
#    f = open("scores.csv", 'w')
#    writer = csv.writer(f)
#    writer.writerow(['id, label'])
#    a = 
#    writer.writerow([label[i], prediction[i]])
#    f.close()