# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 15:41:02 2018

@author: Danny
"""

#!usr/bin/env python  
#-*- coding: utf-8 -*-  

import time  
from sklearn import metrics  
from sklearn.decomposition import PCA
import numpy as np  
import pickle as pickle

import matplotlib.pyplot as plt
from matplotlib import cm
  
# Multinomial Naive Bayes Classifier  
def naive_bayes_classifier(train_x, train_y):  
    from sklearn.naive_bayes import MultinomialNB  
    model = MultinomialNB(alpha=0.01)  
    model.fit(train_x, train_y)  
    return model  
  
  
# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_x, train_y)  
    return model  
  
  
# Logistic Regression Classifier  
def logistic_regression_classifier(train_x, train_y):  
    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression(penalty='l2')  
    model.fit(train_x, train_y)  
    return model  
  
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=8)  
    model.fit(train_x, train_y)  
    return model  
  
  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier(n_estimators=200)  
    model.fit(train_x, train_y)  
    return model  
  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  
  
# SVM Classifier using cross validation  
def svm_cross_validation(train_x, train_y):  
    from sklearn.grid_search import GridSearchCV  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)  
    grid_search.fit(train_x, train_y)  
    best_parameters = grid_search.best_estimator_.get_params()  
    for para, val in best_parameters.items():  
        print (para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)  
    model.fit(train_x, train_y)  
    return model  
def mlp_classifier(train_x,train_y):
    from sklearn.neural_network import MLPClassifier
    model =  MLPClassifier(hidden_layer_sizes=(100,), max_iter=15, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
    model.fit(train_x,train_y)
    return model

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
    thresh = 0.5  
    model_save_file = None  
    model_save = {}  
    print ('reading training and testing data...')
    train_x, train_y, test_x, test_y = read_data(data_file)  
    num_train, num_feat = train_x.shape  
    num_test, num_feat = test_x.shape  
    is_binary_class = (len(np.unique(train_y)) == 2)  
    print ('******************** Data Info *********************')
    print ('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))
#plot training data
     #  select a starting point to display the predictions        
    startindex = 100
    fig = plt.figure(1)
    for index in range(16):
        plt.subplot(4, 4, index+1)
        plt.axis('off')
        plt.imshow(test_x[startindex+index+1,:].reshape(28,28), cmap=cm.binary)
    plt.show()
       
#choose your favorate machine learning algorithem
    test_classifiers = ['NN']
    classifiers = {'NB':naive_bayes_classifier,   
                   'KNN' :knn_classifier, 
                   'LR':logistic_regression_classifier,  
                   'RF':random_forest_classifier,  
                   'DT':decision_tree_classifier,  
                   'SVM':svm_classifier,  
                   'SVMCV':svm_cross_validation,  
                   'GBDT':gradient_boosting_classifier,
                   'NN':mlp_classifier
    }  
      
    
              
      
    for classifier in test_classifiers:  
        print ('******************* %s ********************' % classifier)
        start_time = time.time()  
        model = classifiers[classifier](train_x, train_y)  
       
        predict = model.predict(test_x)  
        if model_save_file != None:  
            model_save[classifier] = model  
        if is_binary_class:  
            precision = metrics.precision_score(test_y, predict)  
            recall = metrics.recall_score(test_y, predict)  
            print ('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        print ('training took %fs!' % (time.time() - start_time) )    
        accuracy = metrics.accuracy_score(test_y, predict)  
        print ('accuracy: %.2f%%' % (100 * accuracy))
  
    if model_save_file != None:  
        pickle.dump(model_save, open(model_save_file, 'wb'))  

    fig = plt.figure(2)
    for index in range(16):
        plt.subplot(4, 4, index+1)
        plt.axis('off')
        plt.imshow(test_x[startindex+index+1,:].reshape(28,28), cmap=cm.binary)
        plt.title('Prediction: %i' % test_y[startindex +index+1])
