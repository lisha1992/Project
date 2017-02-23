#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 19:32:02 2017

@author: ceciliaLee
"""

from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model
from sklearn.cross_validation import KFold, cross_val_score
import time
from sklearn import random_projection

## kNN Classifier
def knn_classifier(train_label, train_data):
    print '==============='
    print 'kNN Classifier: '
    
    clf_knn = KNeighborsClassifier(n_neighbors=5)
 #   scores = cross_val_score(clf_knn, train_data, train_label, cv=10)
    
    k_fold = KFold(len(train_label), n_folds=10, shuffle=True, random_state=0)
    scores = cross_val_score(clf_knn, train_data, train_label, cv=k_fold, n_jobs=1)
    print("Accuracy of KNN Classifier: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print '10-fold Cross Validation Scores:', scores


### Naive Bayes
def guassianNB_classifier(train_label, train_data): 
    print '==============='
    print 'Guassian Naive Bayes Classifier'
   
    clf_gnb = GaussianNB()
    k_fold = KFold(len(train_label), n_folds=10, shuffle=True, random_state=0)
    scores = cross_val_score(clf_gnb, train_data, train_label, cv=k_fold, n_jobs=1)
       
    print("Accuracy of Guassian Naive Bayes Classifier: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    print '10-fold Cross Validation Scores:', scores        
    

def MultinomialNB_classifier(train_label, train_data):
    print '==============='
    print 'Multinomial Naive Bayes Classifier'    
    
    clf_mnb = MultinomialNB()
    k_fold = KFold(len(train_label), n_folds=10, shuffle=True, random_state=0)
    scores = cross_val_score(clf_mnb, train_data, train_label, cv=k_fold, n_jobs=1)
   
    print("Accuracy of Multinomial Naive Bayes Classifier: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    print '10-fold Cross Validation Scores:', scores        
    

### Deciion Trees
def decision_tree_classifier(train_label, train_data):
    print '==============='
    print 'Decision Trees Classifier' 
    
    clf_dt = tree.DecisionTreeClassifier()
    k_fold = KFold(len(train_label), n_folds=10, shuffle=True, random_state=0)
    scores = cross_val_score(clf_dt, train_data, train_label, cv=k_fold, n_jobs=1)
   
    print("Accuracy of Decision Trees Classifier: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    print '10-fold Cross Validation Scores:', scores        
    

### Random Forest
def random_forest_classifier(train_label, train_data):
    print '==============='
    print 'Random Forest Classifier' 
    
    clf_rf = RandomForestClassifier(n_estimators=100)
    k_fold = KFold(len(train_label), n_folds=10, shuffle=True, random_state=0)
    scores = cross_val_score(clf_rf, train_data, train_label, cv=k_fold, n_jobs=1)
  
    print("Accuracy of Random Forest Classifier: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    print '10-fold Cross Validation Scores:', scores        
    
    
## SGD
def stochastic_gradient_descenbt_classifier(train_label, train_data):
    print '==============='
    print ' Stochastic Gradient Descent Classifier' 
    
    clf_sgd = SGDClassifier()
    k_fold = KFold(len(train_label), n_folds=10, shuffle=True, random_state=0)
    scores = cross_val_score(clf_sgd, train_data, train_label, cv=k_fold, n_jobs=1)
    
    print("Accuracy of SGD Classifier: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    print '10-fold Cross Validation Scores:', scores        
    


### SVM
def suport_vector_machine_classifier(train_label, train_data):
    ## LinearSVC implements “one-vs-the-rest” multi-class strategy
    print '==============='
    print ' Linear SVM (one-vs-the-rest) Classifier' 
    
    clf_Linsvm = LinearSVC()
    k_fold = KFold(len(train_label), n_folds=10, shuffle=True, random_state=0)
    scores = cross_val_score(clf_Linsvm, train_data, train_label, cv=k_fold, n_jobs=1)
      
    print("Accuracy of Linear SVM Classifier: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    print '10-fold Cross Validation Scores:', scores        
   

def SVC(train_label, train_data):
     ## SVC implements “one-against-one” multi-class strategy
    print '==============='
    print 'SVM (one-against-one) Classifier' 
   
    clf_svm = svm.SVC(decision_function_shape='ovo')
    k_fold = KFold(len(train_label), n_folds=10, shuffle=True, random_state=0)
    scores = cross_val_score(clf_svm, train_data, train_label, cv=k_fold, n_jobs=1)
    
    print("Accuracy of SVM Classifier: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    print '10-fold Cross Validation Scores:', scores        

    
### Multi-class Logistic Regression
def multi_logistic_reg(train_label, train_data):
    print '==============='
    print 'Multi-Class Logistic Regression Classifier'
    clf_mlr = linear_model.LogisticRegression(multi_class = 'multinomial',solver='lbfgs')
    
    k_fold = KFold(len(train_label), n_folds=10, shuffle=True, random_state=0)
    scores = cross_val_score(clf_mlr, train_data, train_label, cv=k_fold, n_jobs=1)
    
    print("Accuracy of Multi-Class Logistic Regression Classifier: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    print '10-fold Cross Validation Scores:', scores        
    
    

def dimRed_random_projection(x):
    transformer = random_projection.SparseRandomProjection()
    x_new = transformer.fit_transform(x)
    print 'x_new.shape', x_new.shape
    return x_new

def label_mapping():
    f = open('/Users/ceciliaLee/Desktop/Windows/csv/malware_naming.csv')
    classes=[]
    map_dict = {}

    for i in f.readlines()[1:]:
        classes.append(i.split(',')[0])
        index=[i for i in range(len(set(classes)))]
        
    for i in range(len(set(classes))):
        map_dict[list(set(classes))[i]]=index[i]

  #  print len(classes)  # 17267
  #  print len(set(classes)) # 40 
    return map_dict

def coarse_grain(mal_type):
    label = []
    for i in mal_type:
        if 'Worm' in i:
            i=1
            
        ## Trojans
        elif 'Trojan' in i:
            i=2
        elif 'Backdoor' in i:
            i=2
        elif 'Rootkit' in i:
            i=2
        elif 'Exploit' in i:
            i=2
            
        ## Virus    
        elif 'Virus' in i:
            i=3
        
        ## Suspicious packers
        elif 'Packed' in i:
            i=4
        
        ## Malicious Tools
        elif 'HackTool' in i:
            i=5
        elif 'Hoax' in i:
            i=5
        elif 'Flooder' in i:
            i=5
        elif 'Constructor' in i:
            i=5
        elif 'Spoofer' in i:
            i=5
        elif 'VirTool' in i:
            i=5
        
        ## PornWare
        elif 'not-a-virus:Porn-Dialer' in i:
            i=6
            
        ## Adware
        elif 'not-a-virus:AdWare' in i:
            i=7
        ## RiskWare
            
        elif 'not-a-virus' in i:
            i=8

        elif 'DangerousObject' in i:
            i=9
        
        else:
            i=10
        label.append(i)
        
    return label




if __name__ == '__main__':
    map_dict = label_mapping()
  #  print map_dict
    train_dataset = genfromtxt(open('/Users/ceciliaLee/Desktop/Windows/csv/API_Frequency.csv','r'), delimiter=',', dtype='f8')[1:]
    print 'Data finished loading.'
    train_data = [x[2:] for x in train_dataset]

    train_label =[]
#    mal_type =[]
    f = open('/Users/ceciliaLee/Desktop/Windows/csv/API_Frequency.csv')
    for i in f.readlines()[1:]:      
        train_label.append(map_dict[i.split(',')[0].split('.')[0].replace('HEUR:', '')])
     #   mal_type.append(i.split(',')[0])
  #  train_label=coarse_grain(mal_type)
        
    print set(train_label)
    print len(train_label)
    print train_data[0]
    print len(train_data)

  
    knn_classifier(train_label, train_data)
    guassianNB_classifier(train_label, train_data)
    MultinomialNB_classifier(train_label, train_data)
    decision_tree_classifier(train_label, train_data)
    random_forest_classifier(train_label, train_data)
    stochastic_gradient_descenbt_classifier(train_label, train_data)
    suport_vector_machine_classifier(train_label, train_data)
    SVC(train_label, train_data)
    multi_logistic_reg(train_label, train_data)
    
    
    
    
 

    
    
    