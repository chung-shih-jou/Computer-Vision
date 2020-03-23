# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:19:55 2018

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


##############################
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
#            if number%7==0:
#                im.save("train_"+str(number/7)+".png")
        meanface/=number
        
        meanfaceimage = Image.fromarray(np.ceil(meanface))
        meanfaceimage = meanfaceimage.convert("L")
#        meanfaceimage.save("meanface.png")
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
#            im.save("test_"+str(number/7)+".png")
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
        covVects = np.dot(difference.T, eigVects)
        covVects_orth = scipy.linalg.orth(np.dot(difference.T, eigVects))
        for i in range(5):
            a = Image.fromarray(np.reshape(np.real(covVects[:,i]),(56,46)))
#            a.show()
            image = a.convert("L")
#            image.save("eigenface_"+str(i)+".png")
        return covVects_orth,difference,covVects,eigVect
    else:
        return difference
    
def mu(image_list,data_class,train_number):
    ave = np.zeros((1,width*height), dtype=np.float)
    ave_total = np.zeros((1,width*height), dtype=np.float)
    mu = np.zeros((data_class,width*height), dtype=np.float)
    for i in range(data_class*train_number):
        ave += image_list[i,:]
        ave_total += image_list[i,:]
        if i%train_number == 0 and i !=0:
            mu[int(i/train_number)-1,:] = ave/train_number
            ave =np.zeros((1,width*height), dtype=np.float)
    mu[data_class-1,:] = ave/train_number
    ave_total /= train_number*data_class
    a = Image.fromarray(np.reshape(ave_total,(width,height)))
#    a.show()
    return mu, ave_total

def Sb(TrainImage,TrainMu, ave_total,data_class,train_number,covVects):
    Sb = np.zeros((width*height,width*height), dtype=np.float)
    for i in range(data_class):
        A = TrainMu[i,:]-ave_total
        Sb += np.dot(A.T,A)
    Sb *= train_number
    Sw = np.zeros((width*height,width*height), dtype=np.float)
    for i in range(train_number*data_class):
        j = i//train_number
        A =TrainImage[i,:] - TrainMu[j,:]
        Sw += np.dot(A.T,A)
    
    Sbb = np.dot(covVects.T,Sb)
    Sbb = np.dot(Sbb,covVects)
    Sww = np.dot(covVects.T,Sw)
    Sww = np.dot(Sww,covVects)
    eigval,eigVect = linalg.eig(Sb)
    eigvals,eigVects = linalg.eig(Sbb)
    f = np.dot(covVects,eigVects)
    
    for i in range(5): ####可調整'5'來plot出想要的fisherface數量
        a = Image.fromarray(np.reshape(np.real(f[:,i]),(width,height)))
#        a.show()
        image = a.convert("L")
        image.save("fisherface_"+str(i)+".png")
    return eigVect
        
        
def t_SNE(b):
    a = man.TSNE(n_components=2, random_state=0).fit_transform(b[:,0:30])
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
    path_test = 'C:/Users/COSH/Downloads/研究所/上課內容/Computer Vision/作業二/新增資料夾 (2)/*'
    
    im_r,meanface_r,number_r = imagelist_meanface(path_train,0)
    im_t,number_t = imagelist_meanface(path_test,1)
    difference_t = difference_and_eigenface(im_t,meanface_r,number_t,1)
### LDA    
#    clf = nei.KNeighborsClassifier(1).fit(im_r,y)
    
        
    data_class = 40
    width = 56
    height = 46
    im_r,number_r = imagelist_meanface(path_train,1)
    FaceVects,difference_r,covVects,eigVect = difference_and_eigenface(im_r,meanface_r,number_r,0)
    TrainMu, ave_total = mu(im_r,data_class,train_number)
    eigVect = Sb(im_r,TrainMu, ave_total,data_class,train_number,covVects)
    b = np.dot(im_t,eigVect)
    t_SNE(b)
    ### LDA 