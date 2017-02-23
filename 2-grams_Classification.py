#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:37:48 2017

@author: ceciliaLee
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:05:07 2017

@author: ceciliaLee
"""
import pandas as pd
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

def label_mapping():
    f = open('/Users/ceciliaLee/Desktop/Windows/csv/malware_naming.csv')
    classes=[]
    map_dict = {}

    for i in f.readlines()[1:]:
        classes.append(i.split(',')[0])
        index=[i for i in range(len(set(classes)))]
        
    for i in range(len(set(classes))):
        map_dict[list(set(classes))[i]]=index[i]
        
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


if __name__ == '__main__':

    f=open('/Users/ceciliaLee/Desktop/Windows/csv/labeled.csv')
    a=[]
    sample_list_old=[]   # len=17267
    classes=[]
    fine_grain_label_dict = {}

    #### labeling
    for i in f.readlines():
        a.append(i.strip('\r\n').split(','))
    for i in a:
        new_list = [ x for x in i if x != '']
        sample_list_old.append(new_list[2:])  
        classes.append(i[0].replace('HEUR:', '').split('.')[0])
  #  print sample_list[0]    no category and sha
    f.close()
    
    sample_list = [] # sample_list[0]:label, sample_list[1]: sha, sample_list[2]:list    
    for i in sample_list_old:      
        pro_api = process_api(i)
        sample_list.append(reduce_con_api(pro_api))
    print len(sample_list)    
 #   print sample_list[0]             
                      
    index=[i for i in range(len(set(classes)))]
    for i in range(len(set(classes))):
        fine_grain_label_dict[list(set(classes))[i]]=index[i]
   # print classes[0] = 'Worm'
    
  #  print fine_grain_label_dict
  #  fine_grain = dict((k, v) for v, k in fine_grain_label_dict.items())
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
    
    ###### 2-grams of each sample
    ngrams_list = []  # len=17267
    for i in sample_list:
        ngrams_list.append(list(extrac_ngrams(i, 2)))
 
    all_ngrams = []
    for i in ngrams_list:
        for j in i:
            all_ngrams.append(j)            
    all_uni_ngrams=list(set(all_ngrams))    ## 30013 unique 2-grams
# print len(all_ngrams) 4403801
    print len(all_ngrams)
    print len(all_uni_ngrams)
 
       end_counts = []
    for i in ngrams_list:
        count = {}
        for j in all_uni_ngrams:
            count[j] = 0
        for k in i:
         #   if k in all_uni_ngrams:
             count[k]+=1
        end_counts.append(count)
    
    train_label = []
    for i in classes:
        train_label.append(fine_grain_label_dict[i])
        

    print train_label[0]
    print len(train_label)
    print end_counts[0]
    
    train_data_df = pd.DataFrame(end_counts)

    print len(end_counts)
    print len(end_counts[0])
    train_data = dimRed_random_projection(train_data_df.values)
    
    

    coarse_train_label = []
    for i in classes:
        for key, val in coarse_grain.items():
            if i in val:
                coarse_train_label.append(key)
    print len(coarse_train_label)


    ########## Classification ##########
    knn_classifier(coarse_train_label, train_data)
    guassianNB_classifier(coarse_train_label, train_data)
    MultinomialNB_classifier(coarse_train_label, train_data)
    decision_tree_classifier(coarse_train_label, train_data)
    random_forest_classifier(coarse_train_label, train_data)
    stochastic_gradient_descenbt_classifier(coarse_train_label, train_data)
    suport_vector_machine_classifier(coarse_train_label, train_data)
    SVC(coarse_train_label, train_data)
    multi_logistic_reg(coarse_train_label, train_data)
    
    ####################
    knn_classifier(train_label, train_data)
    guassianNB_classifier(train_label, train_data)
    MultinomialNB_classifier(train_label, train_data)
    decision_tree_classifier(train_label, train_data)
    random_forest_classifier(train_label, train_data)
    stochastic_gradient_descenbt_classifier(train_label, train_data)
    suport_vector_machine_classifier(train_label, train_data)
    SVC(train_label, train_data)
    multi_logistic_reg(train_label, train_data)
    
    


    
                
    
     
        