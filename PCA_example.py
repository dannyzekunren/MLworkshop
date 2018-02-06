# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 23:41:37 2018

@author: Danny
"""
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle as pickle

def read_data(data_file):  
    import gzip  
    f = gzip.open(data_file, "rb")
    Myunpickle = pickle._Unpickler(file = f, fix_imports=True,
    encoding="bytes", errors="strict")
    #train,val,test=pickle.load(f)
    train,val,test = Myunpickle.load()
    f.close()  
    train_x = train[0]  
    train_y = train[1]  
    test_x = test[0]  
    test_y = test[1]  
    return train_x, train_y, test_x, test_y  
      
if __name__ == '__main__':  
    data_file = "mnist.pkl.gz" 
    train_x, train_y, test_x, test_y = read_data(data_file)
 ## PCA example       
    pca = PCA(n_components=3)# specify PCA components
    print(pca)
    pca.fit(train_x)
    print(pca.explained_variance_ratio_)
    X_transformed = pca.fit_transform(train_x)
    
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    X, Y, Z = X_transformed[0:200, 0], X_transformed[0:200, 1], X_transformed[0:200, 2]
    test =  np.int8(train_y[:200])
    for x, y, z, s in zip(X, Y, Z, test): 
       c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
    plt.show()