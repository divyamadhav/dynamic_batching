#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import argmax
from numpy import sqrt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.utils import resample
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from statistics import median
import pickle
import csv
import multiprocess
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("../")
from projects import project_list


project_list = ['junit-team-junit5/junit-team-junit5.csv', 'ratpack-ratpack/ratpack-ratpack.csv', 'p6spy-p6spy/p6spy-p6spy.csv']
# In[2]:


def get_first_failures(df):
    
    results = df['tr_status'].tolist()
    length = len(results)
    verdict = ['keep']
    prev = results[0]
    
    for i in range(1, length):
        if results[i] == 0:
            if prev == 0:
                verdict.append('discard')
                #print(i+1)
            else:
                verdict.append('keep')
        else:
            verdict.append('keep')
        prev = results[i]
    
    df['verdict'] = verdict
    df = df[ df['verdict'] == 'keep' ]
    df.drop('verdict', inplace=True, axis=1)
    return df


# In[3]:


folder_path = '../data/25_1_travis_data/'
write_path = 'datasets/'


# In[6]:


for p in project_list:
    
    p_name = p.split('/')[1]
    data = pd.read_csv(folder_path + p)
    data = get_first_failures(data)
    sec = len(data)//10
    
    for i in range(9):
        sec_data = data.iloc[i*sec:i*sec+sec]
        b_ids = sec_data['tr_build_id'].tolist()
        filename = write_path + p_name + '_' + str(i) + '.pkl'
        with open(filename, 'wb') as f:
                pickle.dump(b_ids, f)
        print(p, i, len(b_ids))
        
    
    i = i+1
    sec_data = data.iloc[i*sec:]
    filename = write_path + p_name + '_' + str(i) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(b_ids, f)
    b_ids = sec_data['tr_build_id'].tolist()
    print(p, i, len(b_ids))
    print('\n\n')
        


# In[5]:


data


# In[ ]:




