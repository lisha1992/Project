# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:37:07 2017

@author: ceciliaLee
"""

import csv
import pandas as pd



f=open('/Users/ceciliaLee/Desktop/Windows/dataset/malware_API_dataset.csv')


########## Divide all records into labeled and unlabeld samples
a=[]
b=[]
c=[]
labeled = []  # len(labeled) = 17268
unlabeled = [] # len(unlabeled) = 5878
for i in f.readlines():
    a.append(i)   
for i in a:
    b.append(i.strip('\r\n').split(','))
for i in b:
    c.append([ii.strip('"') for ii in i])

for i in c:
    if len(i[0])!=0:
        labeled.append(i)
    else:
        unlabeled.append(i)
"""
## Write labeled and unlabeld records into 2 csv
with open('labled.csv','w') as f:
    for sublist in labeled:
        for item in sublist:
            f.write(item + ',')
        f.write('\n')        
with open('unlabled.csv','w') as f:
    for sublist in unlabeled:
        for item in sublist:
            f.write(item + ',')
        f.write('\n')      
"""
########## Divide all records into labeled and unlabeld samples
######### ===================================================  #########    

all_family=[]
counter = []   # api frequency for all reconds
new_rec= []  # each element contains the family label, sha, and API frequency

for sample in labeled:
    all_family.append(sample[0])
    
    count = {}
    rec=[]
    rec.append(sample[0])  # family 
    rec.append(sample[1])  # sha

    for api in sample[2:]:
        if len(api) !=0:
            if not count.get(api):
                count[api] = 1
            else:
                count[api] += 1
    count['family']=sample[0]
    count['sha']=sample[1]
        
    rec.append(count)
    new_rec.append(rec)
    counter.append(count)
    
all_api = []  ## All API, 1138 in total
for i in counter:
    for key in i.keys():
        if key not in all_api:
            all_api.append(key)
 #   print set(all_family)
   # print len(set(all_family))
    
for rec in counter:  # rec[0]:family, rec[1]:sha, rec[2]:API frequency
    for i in all_api:
        if i not in rec.keys():
            rec[i]=0
  #  print new_rec[0:2]
 #   print counter[0]
 #   print len(counter)
df = pd.DataFrame(counter)
df.to_csv('API_Frequency.csv')
print df
    
        
        