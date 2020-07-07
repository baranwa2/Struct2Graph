# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:12:24 2020

@author: mayank
"""

import os
from os import walk

import numpy as np
from random import shuffle


filepath = "pdb_files/input1/"

for (dirpath, dirnames, filenames) in walk(filepath):
    break

os.chdir(filepath)
        
for word in filenames[:]:
    if not word.startswith('names'):
        filenames.remove(word)
        
p_list, m_list, s_list = [], [], []
for f in filenames:
    a      = np.load(f)
    st_ind = int(f.strip().split('_')[1])
    
    for e, p in enumerate(a):
        p_list.append(p)
        m_list.append(st_ind)
        s_list.append(e)
    

filepath = "../../"
os.chdir(filepath)

with open('interactions_data.txt', 'r') as f:
    protein_list = f.read().strip().split('\n')

p1_list, p2_list, int_list = [], [], []
i1_list, i2_list           = [], []
for pair in protein_list:
    p1, p2, int_val  = pair.strip().split('\t')
    
    if p1 in p_list and p2 in p_list:
        p1_list.append(p1)
        p2_list.append(p2)
        int_list.append(int(int_val))
        
        i1 = p_list.index(p1)
        i2 = p_list.index(p2)
        
        i1_list.append(m_list[i1]+s_list[i1]-1)
        i2_list.append(m_list[i2]+s_list[i2]-1)
        
### Create Examples
filepath = "pdb_files/input1/"
os.chdir(filepath)

merge_examples = []

for e, _ in enumerate(p1_list):
    merge_examples.append([i1_list[e], i2_list[e], int_list[e]])

shuffle(merge_examples)
shuffle(merge_examples)
shuffle(merge_examples)

print('Total number of examples : ' + str(len(merge_examples)))


outF = open("all_examples.txt", "w")
for line in merge_examples:
    outF.write(str(line[0]))
    outF.write("\t")
    outF.write(str(line[1]))
    outF.write("\t")
    outF.write(str(line[2]))
    outF.write("\n")
outF.close()

np.save('all_examples',merge_examples)

### Create train, test and dev examples

dataset_train_len = int(np.floor(0.8*len(merge_examples)))
dataset_rem_len   = len(merge_examples)-dataset_train_len
dataset_test_len  = int(np.floor(0.5*dataset_rem_len))

dataset_train = []
for i in range(dataset_train_len):
    dataset_train.append(merge_examples[i])
    
dataset_test = []
for i in range(dataset_train_len,dataset_train_len+dataset_test_len):
    dataset_test.append(merge_examples[i])
    
dataset_dev = []
for i in range(dataset_train_len+dataset_test_len,len(merge_examples)):
    dataset_dev.append(merge_examples[i])
    
    
np.save('train_examples',dataset_train)
np.save('one_to_one_test_examples',dataset_test)
np.save('one_to_one_dev_examples',dataset_dev)


ind_pos = []
ind_neg = []
for no, d in enumerate(dataset_train):
    if d[-1] == 1:
        ind_pos.append(no)
    else:
        ind_neg.append(no)
    
pos_example = len(ind_pos)



#1:2 Ratio
pos_inds = list(np.random.choice(np.array(ind_pos), int(pos_example/2), replace=False))

dataset_train_12 = []
for no, d in enumerate(dataset_train):
    if no in pos_inds or no in ind_neg:
        dataset_train_12.append(d)
        
shuffle(dataset_train_12)
np.save('train_examples_one_two',dataset_train_12)



#1:3 ratio
pos_inds = list(np.random.choice(np.array(ind_pos), int(pos_example/3), replace=False))

dataset_train_13 = []
for no, d in enumerate(dataset_train):
    if no in pos_inds or no in ind_neg:
        dataset_train_13.append(d)
        
shuffle(dataset_train_13)
np.save('train_examples_one_three',dataset_train_13)


#1:5 ratio
pos_inds = list(np.random.choice(np.array(ind_pos), int(pos_example/5), replace=False))

dataset_train_15 = []
for no, d in enumerate(dataset_train):
    if no in pos_inds or no in ind_neg:
        dataset_train_15.append(d)
        
shuffle(dataset_train_15)
np.save('train_examples_one_five',dataset_train_15)


#1:10 ratio
pos_inds = list(np.random.choice(np.array(ind_pos), int(pos_example/10), replace=False))

dataset_train_110 = []
for no, d in enumerate(dataset_train):
    if no in pos_inds or no in ind_neg:
        dataset_train_110.append(d)
        
shuffle(dataset_train_110)
np.save('train_examples_one_ten',dataset_train_110)