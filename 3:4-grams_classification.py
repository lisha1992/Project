#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:05:07 2017

@author: ceciliaLee
"""
import pandas as pd
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model
from sklearn.cross_validation import KFold, cross_val_score

from sklearn import random_projection

def extrac_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])
def process_api(api_list):
    new_api_list = []
    for i in api_list:
        if i.endswith('ExA'):
            new_i = i.rstrip('ExA')
        elif i.endswith('ExW'):
            new_i = i.rstrip('ExW')
        elif i.endswith('Ex'):
            new_i = i.rstrip('Ex')
        elif i.endswith('A'):
            new_i = i.rstrip('A')
        elif i.endswith('W'):
            new_i = i.rstrip('W')
        else:
            new_i = i
        new_api_list.append(new_i)
    return new_api_list

def reduce_con_api(api_list):
    tmp = []
    i=1
    while i <len(api_list):
        if api_list[i]!=api_list[i-1]:
            tmp.append(api_list[i])
            i+=1
        else:
            i+=1
    return tmp   

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

def counter(sample_list, sample_ngrams):
    counts = []
    n = 0
    for i in sample_list:
        count_dict = {}
        for j in sample_ngrams:
            count_dict.setdefault(j, 0)
        for k in i:
            if k in sample_ngrams:
                count_dict[k] +=1
        counts.append(count_dict)
        print n
        n+=1
    return counts

def job1(fine_train_label, train_data):
    multi_logistic_reg(fine_train_label, train_data)
    guassianNB_classifier(fine_train_label, train_data)
    knn_classifier(fine_train_label, train_data)
    decision_tree_classifier(fine_train_label, train_data)
    random_forest_classifier(fine_train_label, train_data)
    stochastic_gradient_descenbt_classifier(fine_train_label, train_data)
    suport_vector_machine_classifier(fine_train_label, train_data)
    SVC(fine_train_label, train_data)
    
    
def job2(coarse_train_label, train_data):
  #  multi_logistic_reg(coarse_train_label, train_data)
  #  guassianNB_classifier(coarse_train_label, train_data)
  #  knn_classifier(coarse_train_label, train_data)
  #  decision_tree_classifier(coarse_train_label, train_data)
  #  random_forest_classifier(coarse_train_label, train_data)
  #  stochastic_gradient_descenbt_classifier(coarse_train_label, train_data)
    suport_vector_machine_classifier(coarse_train_label, train_data)   
    SVC(coarse_train_label, train_data)
                       
if __name__ == '__main__':

    f=open('/Users/ceciliaLee/Desktop/Windows/csv/labeled.csv')
    a=[]
    
    classes = []
    sample_list=[]   # len=17267
    for i in f.readlines():
        a.append(i.strip('\r\n').split(','))
 #   print a[0]
    for i in a:
        new_list = [ x for x in i if x != '']
        classes.append(new_list[0].replace('HEUR:', '').split('.')[0])
        sample = reduce_con_api(extrac_ngrams(process_api(new_list[2:]),4))
        sample_list.append(sample)
    f.close()
    print 'len(sample_list)', len(sample_list)
    print 'len(classes)', len(classes)
    
  ########## Train label ########## 
    fine_grain={}                 
    index=[i for i in range(len(set(classes)))]
    for i in range(len(set(classes))):
        fine_grain[list(set(classes))[i]]=index[i]
    print len(fine_grain)
    coarse_grain = {0: ['Worm','Email-Worm', 'P2P-Worm', 'IM-Worm','Net-Worm' ], 
                    1: ['Trojan-FakeAV', 'Trojan-PSW','Trojan-GameThief','Trojan-Clicker', 
                    'Trojan', 'Trojan-Downloader','Trojan-Ransom','Trojan-Spy','Trojan-Proxy', 
                    'Trojan-Banker', 'Trojan-Dropper','Rootkit','Exploit','Backdoor'],
                    2: ['Virus'],
                    3: ['Packed'],
                    4: ['HackTool', 'Hoax', 'Constructor','Flooder','Spoofer','VirTool'],
                    5: ['not-a-virus:Porn-Dialer'],
                    6: ['not-a-virus:AdWare'],
                    7: ['not-a-virus:Dialer','not-a-virus:Server-Web','not-a-virus:WebToolbar','not-a-virus:Client-IRC',
                        'not-a-virus:NetTool','not-a-virus:Downloader','not-a-virus:Monitor','not-a-virus:RiskTool',
                        'not-a-virus:PSWTool','not-a-virus:Server-Proxy','not-a-virus:RemoteAdmin'],
                    8: ['UDS:DangerousObject']           
                    }
    
    fine_train_label = []
    for i in classes:
        fine_train_label.append(fine_grain[i])
    print 'len(fine_train_label): ', len(fine_train_label)
    coarse_train_label = []
    for i in classes:
        for key, val in coarse_grain.items():
            if i in val:
                coarse_train_label.append(key)
    print 'len(coarse_train_label): ', len(coarse_train_label)
    ########## Train label ##########  
    ########## Train data ########## 
    
   
    sample_ngrams = []
    f=open('/Users/ceciliaLee/Desktop/txt/extracted_top_4grams8000.txt')
    for i in f.readlines():
        sample_ngrams.append(tuple((i.strip('\n').split(' '))))
    print 'len(sample_ngrams)', len(sample_ngrams)
    end_counts = counter(sample_list, sample_ngrams)
    print 'len(end_counts): ', len(end_counts)
    print 'len(end_counts[0]): ', len(end_counts[0])

    with open('/Users/ceciliaLee/Desktop/txt/end_counts_4n_8000_part1.txt', 'w') as f:
        n=0
        for i in end_counts[:5000]:
            for key, value in i.items():
                f.write('%d ' % (value))
            print n
            n+=1
            f.write('\n')
    print 'Output Done part1.'
    with open('/Users/ceciliaLee/Desktop/txt/end_counts_4n_8000_part2.txt', 'w') as f:
        n=5000
        for i in end_counts[5000:10000]:
            for key, value in i.items():
                f.write('%d ' % (value))
            print n
            n+=1
            f.write('\n')
    print 'Output Done part2.'
    with open('/Users/ceciliaLee/Desktop/txt/end_counts_4n_8000_part3.txt', 'w') as f:
        n=10000
        for i in end_counts[10000:15000]:
            for key, value in i.items():
                f.write('%d ' % (value))
            print n 
            n+=1
            f.write('\n')
    print 'Output Done part3.'
    with open('/Users/ceciliaLee/Desktop/txt/end_counts_4n_8000_part4.txt', 'w') as f:
        n=15000
        for i in end_counts[15000:]:
            for key, value in i.items():
                f.write('%d ' % (value))
            print n
            n+=1
            f.write('\n')
    print 'Output Done part4.'
    """
    
  
    f1=open('/Users/ceciliaLee/Desktop/txt/end_count_3n_20000_part1.txt')
    f2 =open('/Users/ceciliaLee/Desktop/txt/end_counts_3n_20000_part2.txt')
    f3=open('/Users/ceciliaLee/Desktop/txt/end_counts_3n_20000_part3.txt')
    f4 =open('/Users/ceciliaLee/Desktop/txt/end_counts_3n_20000_part4.txt')
    a1=[]
    for i in f1.readlines():
        a1.append(i)
    f1.close()
    for i in f2.readlines():
        a1.append(i)
    f2.close()
    for i in f3.readlines():
        a1.append(i)
    f3.close()
    for i in f4.readlines():
        a1.append(i)
    f4.close()
    print 'len(a1): ', len(a1)
    b1=[]
    for i in a1:
        t = i.strip(' \n').split(' ')
        n = [int(x) for x in t]
        b1.append(n)
    

    print 'len(b1):', len(b1)
    print len(b1[0]),len(b1[1]), len(b1[2])
    
                

  #  train_data_df = pd.DataFrame(end_counts)
    train_data = dimRed_random_projection(b1)
  #  train_data= train_data_df.values
   # print train_data.shape
    
    suport_vector_machine_classifier(coarse_train_label, train_data)   
    SVC(coarse_train_label, train_data)
    suport_vector_machine_classifier(fine_train_label, train_data)
    SVC(fine_train_label, train_data)
    
    """
    
    
