# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 17:13:01 2018

@author: COSH
"""

from PIL import Image
import glob 
import numpy as np 
#from scipy.signal import argrelextrema
from numpy import linalg
import scipy
import sklearn.manifold as man
#from sklearn import neighbors as nei
import matplotlib.pyplot as plt



def imagelist_meanface(path,flag):
    image_list = []
    number = 0
    if flag ==0:
        meanface=np.zeros((56,46), dtype=np.float)
        for filename in glob.glob(path):
            im = Image.open(filename)
            a = np.array(np.reshape(im,(1,56*46)))
            meanface += im
            image_list.append(a)
            number+=1
            if number%7==0:
                im.save("train_"+str(number/7)+".png")#### 此為data load進來的順序；可註解掉
        meanface/=number
        
        meanfaceimage = Image.fromarray(np.ceil(meanface))
        meanfaceimage = meanfaceimage.convert("L")
        meanfaceimage.save("meanface.png")#### 此為meanface；可註解掉
        image_list = np.reshape(image_list,(number,56*46))
        meanface = np.reshape(meanface,(1,56*46))
    #    meanface_total = np.reshape(meanface_total,(int(number/train_number),56*46))
        return image_list,meanface,number
    else:
        for filename in glob.glob(path):
            im = Image.open(filename)
            a = np.array(np.reshape(im,(1,56*46)))
            image_list.append(a)
            number+=1
            im.save("test_"+str(number/7)+".png")#### 此為data load進來的順序；可註解掉
        image_list = np.reshape(image_list,(number,56*46))
        return image_list,number


def difference_and_eigenface(im,meanface,n,flag):
    
    difference = np.zeros((n,46*56), dtype=np.float)
    for x in range(n):
        difference[x,:] = im[x,:]-meanface[0,:]
    if flag ==0:
        i = np.identity(280)
        sigma = np.cov(difference)
        eigval,eigVect = linalg.eig(np.cov(difference.T))
        eigvals,eigVects = linalg.eig(sigma)
        eigSortIndex = np.argsort(-eigvals)
        eigvals = i*eigvals[eigSortIndex]
        eigVects = eigVects[eigSortIndex]
        
        covVects = np.dot(difference.T, eigVects)
        covVects_orth = scipy.linalg.orth(np.dot(difference.T, eigVects))
        for i in range(5):
            a = Image.fromarray(np.reshape(np.real(covVects[:,i]),(56,46)))
#            a.show()
            image = a.convert("L")
            image.save("eigenface_"+str(i)+".png")#### 此為前五個eigenface；可註解掉
        return covVects_orth,difference,eigVect
    else:
        return difference


def compare(difference_t,FaceVects,number_t,meanface_r,eigenface_n,im_t):
    MSE = 0
    weight = np.dot(difference_t,FaceVects) #1x2576 2576x279 = 279
    for n in range(number_t):
        for i in eigenface_n:
            c = np.dot(weight[0,0:i],FaceVects[:,0:i].T) #1xi ix2576 = 2576
            d = c+meanface_r
            diff = im_t-d
            MSE = np.dot(diff,diff.T)/2576
            e = Image.fromarray(np.reshape(d,(56,46)))
#            e.show()
            image = e.convert("L")
            image.save("eigenvector_"+str(i)+".png")#### 此為投影在test data上的影像
            print("MSE:"+str(MSE)+" eigenvetor:"+str(i))
            
    return MSE

def t_SNE(b):
    a = man.TSNE(n_components=2, random_state=0).fit_transform(b[:,0:100])
    colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'brown', 'silver', 'purple', 'gold']
    a_min, a_max = a.min(0), a.max(0)
    a_norm = (a - a_min) / (a_max - a_min)  # 归一化
    for i in range(40):
        arr = np.array([i]*3)
        if i==0:
            y = arr
        else:
            y = np.r_[y,arr]
        
    
    for i in range(a.shape[0]):
        j=i//3%10
        plt.scatter(a_norm[i, 0], a_norm[i, 1],y[i], color=colors[j])
        
if __name__ == '__main__':
    
    eigenface_n = [5,50,100,279]
    train_number = 7
    path_train = 'C:/Users/COSH/Downloads/研究所/上課內容/Computer Vision/作業二/新增資料夾/*'
    path_test = 'C:/Users/COSH/Downloads/研究所/上課內容/Computer Vision/作業二/新增資料夾 (3)/*'##1*2576
#    
#    
    im_r,meanface_r,number_r = imagelist_meanface(path_train,0)
    FaceVects,difference_r,eigVect = difference_and_eigenface(im_r,meanface_r,number_r,0)
#    #Face 280*2576, difference 280*2576
#    
    im_t,number_t = imagelist_meanface(path_test,1) 
    difference_t = difference_and_eigenface(im_t,meanface_r,number_t,1)
#    # diff 1*2576
    MSE = compare(difference_t,FaceVects,number_t,meanface_r,eigenface_n,im_t)
    
    path_test = 'C:/Users/COSH/Downloads/研究所/上課內容/Computer Vision/作業二/新增資料夾 (2)/*'##120*2576
## t-SNE
    im_t,number_t = imagelist_meanface(path_test,1) 
    b = np.dot(im_t,eigVect)
    t_SNE(np.real(b))
        
        
   