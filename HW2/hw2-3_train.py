# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 20:11:51 2018

@author: COSH
"""

import numpy as np
import matplotlib.pyplot as plt

#from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Conv2D,MaxPooling2D  
from keras.models import Model,Sequential  
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
#from keras.utils import to_categorical
import glob 
from PIL import Image
from keras.utils import np_utils 
#import sklearn.manifold as man
#import tensorflow as tf



def load_image(flag,Nclass):
    
    image_list=[]
    number=0
    for i in range(Nclass):
        if flag == 0:
            path_train = 'C:/Users/COSH/Downloads/研究所/上課內容/Computer Vision/作業二/hw2-3_data/train/class_'+str(i)+'/*'
        else:
            path_train = 'C:/Users/COSH/Downloads/研究所/上課內容/Computer Vision/作業二/hw2-3_data/valid/class_'+str(i)+'/*'
        
        for filename in glob.glob(path_train):
            im = Image.open(filename)
            width,height = im.size
            a = np.array(im)
            image_list.append(a)
            number+=1
    image_list = np.reshape(image_list,(number,width,height))
    if flag == 0:
        print("TrainImage:"+str(image_list.shape))
    else:
        print("TestImage:"+str(image_list.shape))
    return image_list,number,width,height

def CNN(Nclass):
    CNNmodel = Sequential()  
    # Create CN layer 1  
    CNNmodel.add(Conv2D(Nclass,(5,5),padding='same', input_shape=(28,28,1),activation='relu'))  
    # Create Max-Pool 1  
    CNNmodel.add(MaxPooling2D(pool_size=(2,2)))     
    # Create CN layer 2  
    CNNmodel.add(Conv2D(Nclass, (5,5),padding='same', input_shape=(28,28,1), activation='relu'))     
    # Create Max-Pool 2  
    CNNmodel.add(MaxPooling2D(pool_size=(2,2)))        
    # Add Dropout layer  
    CNNmodel.add(Dropout(0.2)) 
    CNNmodel.add(Flatten())  
    CNNmodel.add(Dense(128, activation='relu'))  
    CNNmodel.add(Dropout(0.25))  
    CNNmodel.add(Dense(Nclass, activation='softmax'))  
    CNNmodel.summary()  
    
    return CNNmodel


def show_train_history(history, train, validation,title):  
    plt.plot(history.history[train])  
    plt.plot(history.history[validation])  
    plt.title(title)  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'])  
    plt.show()  
    
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
            
if __name__ == '__main__':
#    path_train = 'C:/Users/COSH/Downloads/研究所/上課內容/Computer Vision/作業二/hw2-3_data/*'
#    path_test = 'C:/Users/COSH/Downloads/研究所/上課內容/Computer Vision/作業二/新增資料夾 (3)/*'
    Nclass = 10
    x_train,Train_n,width,height = load_image(0,Nclass)
    x_valid,Valid_n,width,height = load_image(1,Nclass)
    
    x_train = np.expand_dims(x_train, axis=-1) / 255.
    x_valid = np.expand_dims(x_valid, axis=-1) / 255.
    
    model= CNN(Nclass)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])
    y_train=[]
    y_valid=[]
    for i in range(Nclass):
        arr = np.array([i]*5000)
        arr1 = np.array([i]*1000)
        if i==0:
            y_train = arr
            y_valid = arr1 
        else:
            y_train = np.r_[y_train,arr]
            y_valid = np.r_[y_valid,arr1]
            
    Y_train = np_utils.to_categorical(np.uint8(y_train))  
    Y_valid = np_utils.to_categorical(np.uint8(y_valid))
    
    
    ckpt = ModelCheckpoint('CNN_model_e{epoch:02d}', # CNN_model_e{epoch:02d}_a{val_acc:.4f}
                           monitor='val_acc',
                           save_best_only=False,
                           save_weights_only=True,
                           verbose=1)
    cb = [ckpt]
    
    epochs = 15
    batch_size = 128
    histiory = model.fit(x_train,Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_valid,Y_valid),
                        callbacks=cb,
                        verbose=1)
    
    show_train_history(histiory, 'acc', 'val_acc','accuracy curve')  
    show_train_history(histiory, 'loss', 'val_loss','loss curve')
    
    model.save('CNN_model.h5')
    
    ##### visualize conv layers    
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x_train[10000].reshape(1,28,28,1))
    display_activation(activations, 3, 3, 4)
    ##### visualize conv layers 
    
    
    
#    layer1_reshape = tf.reshape(activations[:, :, :, :], [-1, 14 * 14 * 32])
#    layer1_pca = pca(layer1_reshape.eval(feed_dict ={ x: x_valid}), 50)
#    layer1_tsne = man.tsne(layer1_pca, 2)
#    plot_scatter(layer1_tsne, y_valid, "conv layer1 with tsne")
##    
#    colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'brown', 'silver', 'purple', 'gold']
#    plt.figure(figsize=(12, 9))
#    for i in (np.unique(y_valid)):
#        for j in range(len(x_valid)):
#            if y_train[j] == i:
#                plt.scatter(x_valid[j, 0], x_valid[j, 1], color=colors[i])
#    
#    #This is faster loop and it is used to create legent for our graph
#    for i in (np.unique(y_train)):
#           for j in range(len(x_train)):
#                if y_train[j] == i:
#                    plt.scatter(x_train[j, 0], x_train[j, 1], color=colors[i], label='Image of number {}'.format(i))
#                    break
#    plt.legend()
#    plt.title("t-SNE 2D representation of MNIST dataset")
#    plt.show()
