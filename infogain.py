#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:56:04 2017

@author: ceciliaLee
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 11:44:58 2017

@author: ceciliaLee
"""
import numpy as np
import multiprocessing
import math


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

def matrix_cal(input_list, all_ngrams, classes):
    matrix_01 = np.zeros((len(input_list), len(all_ngrams)), dtype=np.int32) # custom dtype    
    for i in all_ngrams:
        index_c=all_ngrams.index(i)
        for key, val in input_list.iteritems():  
            index_r=classes.index(key)   
            count = 0
            for v in val:
                if i in v:     
                    count+=1
            matrix_01[index_r][index_c] = count
     #   print 'done n-grams: %d. \n' % index_c        
    return matrix_01
    
def information_gain(matrix, n_sample, input_list, classes, all_ngrams):
    IG = [] 
    print'n_sample=', n_sample
    for i in all_ngrams: # index1=all_ngrams.index(i)
        index_c = all_ngrams.index(i)
        p_v1 = 1.0*sum(matrix[:, index_c])/n_sample
        p_v0 = 1-p_v1
        ig = 0
       # print 'index_c=', index_c
       # print 'p_v1= ',p_v1
       # print 'p_v0=', p_v0
        for j in classes:  # index2 = classes.index(i)
            index_r=classes.index(j)
            n_c = len(input_list[j])            
            p_c = float(1.0*n_c/n_sample)
            count = matrix[index_r][index_c]
          #  print 'count = ', count
          #  print 'index_r=', index_r
          #  print 'n_c= ',n_c
          #  print 'p_c=', p_c
            if count == 0:
                temp = 1.0*math.log1p((1.0/(p_v0*p_c)))
                ig+=temp
            elif count == n_c:
                temp = 1.0*math.log1p((1.0/np.float(p_v1*p_c)))
                ig+=temp
            elif count !=0 and count < n_c:
                p_v1_c = 1.0*(count / n_c)
                p_v0_c = 1 - p_v1_c     
              #  print 'p_v1_c =', p_v1_c 
              #  print 'p_v0_c =', p_v0_c 
                
                temp = p_v1_c * math.log1p(float(1.0*p_v1_c/(p_v1*p_c)) )+ p_v0_c * math.log1p(np.float(p_v0_c/(p_v0*p_c)))
                ig += temp        
        IG.append(ig)
    return IG
            


if __name__ == '__main__':

    f=open('/Users/ceciliaLee/Desktop/Windows/csv/labeled.csv')
    a=[]
  #  sample_list_old=[]   # len=17267
    sample_list_old = [] # [0]: label. [1]:n-grams list
    for i in f.readlines():
        a.append(i.strip('\r\n').split(','))
 #   print a[0]
    for i in a:
        new_list = [ x for x in i if x != '']
        sample = [new_list[0].replace('HEUR:', '').split('.')[0]]
        sample.append(reduce_con_api(extrac_ngrams(process_api(new_list[2:]),4)))
        sample_list_old.append(sample)
    f.close()
    sample_list = []
    for i in sample_list_old:
        if i not in sample_list:
            sample_list.append(i)
    
   # print len(sample_list)  9896

    input_list={}
    all_ngrams = []
    for i in sample_list:
        if i[0] not in input_list.keys():
            input_list[i[0]] = [i[1]]
        else:
            input_list[i[0]].append(i[1])
   # print len(input_list['Worm'])  575
   # print input_list['Worm'][:2]

    classes = []
    ngrams = []
    for key, val in input_list.iteritems():
#        input_list[key] = list(set(val))
        classes.append(key)
        for v in val:
            for i in v:
                ngrams.append(i)

   # print classes
   # print len(classes) 40
    all_ngrams=list(set(ngrams))  # 72152 , 118981
    print len(all_ngrams)
  #  print len(ngrams) 2753461
  #  print len(set(ngrams))  
    """

    pool = multiprocessing.Pool(processes=4)
    matrix_res= pool.apply_async(matrix_cal, args=(input_list, all_ngrams, classes)) 
    matrix_01 = matrix_res.get()
    pool.close()
    pool.join()
    print 'Matrix calculation (pool) done.\n'
    print matrix_01
    print matrix_01.shape
    
    np.savetxt('/Users/ceciliaLee/Desktop/Windows/matrix_3ngrams.txt', matrix_01, fmt = '%d')
    """
    matrix_01 = np.loadtxt('/Users/ceciliaLee/Desktop/matrix_4ngrams.txt',  dtype=np.int32)
    print matrix_01.shape
    for i in range(matrix_01.shape[0]):
        for j in range(matrix_01.shape[1]):
            if matrix_01[i,j]<0:
                print matrix_01[i,j]
    
    
    n_sample = len(sample_list)
    info_gain = information_gain(matrix_01, n_sample, input_list, classes, all_ngrams)
    thefile = open('/Users/ceciliaLee/Desktop/information_gain_4grams.txt', 'w')
    for i in info_gain:
        thefile.write("%f\n" % i)
    thefile.close()
        
    
    
   # print info_gain
    print len(info_gain)
    print len(set(info_gain))
    rank_ig = sorted(info_gain, reverse = True)
    
    ig_dict = {}
    for i in range(len(info_gain)):
        ig_dict[all_ngrams[i]] = info_gain[i]
    rank_dict_list = sorted(ig_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    
    rank_ngrams = []    
    for i in rank_dict_list:
        rank_ngrams.append(i[0])
    

    f = open("/Users/ceciliaLee/Desktop/rank_ngrams_4grams.txt",'wb')
    for t in rank_ngrams:
        line = ' '.join(str(x) for x in t)
        f.write(line + '\n')
    f.close()
    
    f = open("/Users/ceciliaLee/Desktop/all_ngrams_4grams.txt",'wb')
    for t in all_ngrams:
        line = ' '.join(str(x) for x in t)
        f.write(line + '\n')
    f.close()


    np.savetxt('/Users/ceciliaLee/Desktop/rank_ig_4grams.txt',rank_ig,fmt = '%f')
    np.savetxt('/Users/ceciliaLee/Desktop/set_rank_ig_4grams.txt',list(set(rank_ig)),fmt = '%f')
    

    
