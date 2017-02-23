# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 04:59:43 2017

@author: ceciliaLee
"""
import numpy as np
import multiprocessing
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


def Markov_Chain(api_list, api_record): # Assign new_api_record to api_record
    l = len(api_list)
    adj_mtr = np.zeros([l, l], dtype = np.float16)

    for j in range(len(i)-1):
        call_1 = i[j]
        call_2 = i[j+1]
        index_1 = api_list.index(call_1)
        index_2 = api_list.index(call_2)
        adj_mtr[index_1][index_2] += 1
    for k in range(l):
        adj_mtr[k] = np.divide(adj_mtr[k], sum(adj_mtr[k]), out=np.zeros_like(adj_mtr[k]), where=(sum(adj_mtr[k]))!=0)
            
       # adj_matrix.append(np.ravel(adj_mtr, order='C'))  # 902*902
    return np.ravel(adj_mtr, order='C')

def open_file(file_name):
    ### Return record[0]:class label, record[1]: sha, record[2:]: API 
    f = open(file_name)
    a=[]
    record=[]
    for i in f.readlines():
        a.append(i.strip('\r\n').split(','))
    for i in a:
        new_list = [x for x in i if x != '']
        record.append(new_list)
    f.close()
    return record
    
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

def dimRed_random_projection(x):
    transformer = random_projection.SparseRandomProjection()
    x_new = transformer.fit_transform(x)
    print 'x_new.shape', x_new.shape
    return x_new


def fur_reduced_api(matr_tran):# assign a list, return a list
    count = 0
    matr_new = []
    for i in matr_tran:
         if i==0:
             count+=1
    if count<matr.shape[0]*0.05:
        matr_new.append(matr_tran)
   # matr_new=np.asarray(matr_new).T
   # print matr_new.shape
    return matr_new
    



if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    
    file_name = '/Users/ceciliaLee/Desktop/Windows/csv/labeled.csv'     
    sample_record =  open_file(file_name)  

    sample_list = []  ## [ [class, sha][API sequence] ]
    for i in sample_record:
        li = [i[0]]
        li.append(i[2:])
        sample_list.append(li)
    
    classes = []
    all_api = []  # 4421068, repeating 
    api_record=[]  ## No class label and sha, [ [API sequence1],[API sequence2], ... ]
    for i in sample_list:
        classes.append(i[0].replace('HEUR:', '').split('.')[0])
        temp = process_api(i[1])
        api_record.append(temp)
        for i in temp:
            all_api.append(i)
    all_api_set  = list(set(all_api))  # 902, unique API 

    ##### Train Label
    fine_grain = {}  # 40
    index=[i for i in range(len(set(classes)))]
    for i in range(len(set(classes))):
        fine_grain[list(set(classes))[i]]=index[i]

    
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
    
    """
    ####### Get the transition probability matrix
    print 'Start transition probability matrix calculation.\n'
    result = []
    for i in new_api_list[:3000]:
        result.append(pool.apply_async(Markov_Chain, (api_list, i)))
    print 'done part 1\n'
    
    for i in new_api_list[3000:6000]:
        result.append(pool.apply_async(Markov_Chain, (api_list, i)))
    print 'done part 2\n'
    for i in new_api_list[6000:]:
        result.append(pool.apply_async(Markov_Chain, (api_list, i)))
    print 'done part3.\n'
    pool.close()
    pool.join()
    print 'pool done.\n'
    
    matr=[]   # transition probability matrix, 9481*(902*902)
    for res in result:
        matr.append(res.get())
    matr=np.asarray(matr)
    print matr.shape
   
    print 'Start reduction.\n'
    n = 902*902/11
    reduced = dimRed_random_projection(matr[:, :n])
    for i in range(1,11):
        res = dimRed_random_projection(matr[:, (n*i):(n*(i+1))])
        reduced = np.hstack((reduced, res))
    print reduced.shape
    
    print 'done reduction.\n'
        
   ############## Training ##############
   
   ## Feature reduction
    
    train_data = reduced 
    
   # train_data = matr
    
    
    ###### Training data labels
    train_label_fine = []
    for i in classes:
        train_label_fine.append(fine_grain[i])
    print set(train_label_fine)
    
    #################### 
    train_label_coarse = []
    for i in classes:
        for key, val in coarse_grain.items():
            if i in val:
                train_label_coarse.append(key)
    print set(train_label_coarse)
    
    
    print 'Start model training.\n'
    #################### 
    pro_1 = multiprocessing.Process(target = job1, args = (train_label_fine, train_data))
    pro_2 = multiprocessing.Process(target = job2, args = (train_label_coarse, train_data))
    
    pro_1.start()
    pro_2.start()
    pro_1.join()
    pro_2.join()
    
    MultinomialNB_classifier(train_label_fine, train_data)
    MultinomialNB_classifier(train_label_coarse, train_data)

    """
   
    