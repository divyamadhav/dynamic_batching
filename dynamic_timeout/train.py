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

def output_values(Y_data):
    Y_t = []
    for e in Y_data:
        if e == 'passed':
            Y_t.append(1)
        else:
            Y_t.append(0) 
    return Y_t



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



def pd_get_train_test_data(file_path, first_failures=True):
    columns = ['tr_build_id', 'git_num_all_built_commits', 'git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_modified', 'tr_status']
    X = pd.read_csv(file_path, usecols = columns)
    X['tr_status'] = output_values(X['tr_status'])
    
    if first_failures:
        X = get_first_failures(X)

    return X




def sbs(path, ver, train=1):

    project_file = "../data/25_1_travis_data/" + path
    project_name = path.split('/')[1]
    #dataset is split using first_failures
    project =  pd_get_train_test_data(project_file, first_failures=False)
    
    pkl_file = 'datasets/' + project_name + '_' + str(ver) + '.pkl'
    with open(pkl_file, 'rb') as load_file:
        test_build_ids = pickle.load(load_file)
    

    X_train = project [ ~project['tr_build_id'].isin(test_build_ids)]
    Y_train = X_train['tr_status'].tolist()
    
    X_test = project [ project['tr_build_id'].isin(test_build_ids)]
    
    if train == 0:
        return X_test
    
    num_feature = 4
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
    
    rf = RandomForestClassifier()
    param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth}
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 0)
    
    best_n_estimators = []
    best_max_depth = []
    
    xsample = X_train.copy()
    ysample = xsample['tr_status'].tolist()
    xsample.drop('tr_status', inplace=True, axis=1)
    xsample.drop('tr_build_id', inplace=True, axis=1) 
    best_f1 = 0
    best_f1_sample = xsample
    best_f1_sample_result = ysample
    best_f1_estimator = 0
    best_thresholds = []
    

    for i in range(100):
        print('Bootstrapping {} for {}'.format(i, project_name))
        
        while True:
            print('Here for {} {}'.format(i, project_name))
            sample_train = resample(X_train, replace=True, n_samples=len(X_train))
            sample_train_result = sample_train['tr_status']

            build_ids = sample_train['tr_build_id'].tolist()
            sample_test = X_train [~X_train['tr_build_id'].isin(build_ids)] 
            sample_test_result = sample_test['tr_status']

            if len(sample_test_result) != 0:
                break
        
        sample_train.drop('tr_status', inplace=True, axis=1)
        sample_train.drop('tr_build_id', inplace=True, axis=1)
        sample_test.drop('tr_status', inplace=True, axis=1)
        sample_test.drop('tr_build_id', inplace=True, axis=1)
        
        print('Training {} for {}'.format(i, project_name))
        grid_search.fit(sample_train, sample_train_result)
        sample_pred_vals = grid_search.predict_proba(sample_test)

        pred_vals = sample_pred_vals[:, 1]
        fpr, tpr, t = roc_curve(sample_test_result, pred_vals)
        gmeans = sqrt(tpr * (1-fpr))
        ix = argmax(gmeans)
        bt = t[ix]
        best_thresholds.append(bt)
        
        final_pred_result = []
        #threshold setting
        for j in range(len(pred_vals)):
            if pred_vals[j] > bt:
                final_pred_result.append(1)
            else:
                final_pred_result.append(0)
        
        try:
            f1 = f1_score(sample_test_result, final_pred_result)
        except:
            print('')

        if f1 > best_f1:
            best_f1 = f1
            best_f1_sample = sample_train
            best_f1_sample_result = sample_train_result
            best_f1_estimator = grid_search.best_estimator_
            print(best_f1_sample)
            
        best_n_estimators.append(grid_search.best_params_['n_estimators'])
        best_max_depth.append(grid_search.best_params_['max_depth'])
        
    #completed with bootstrapping 
    threshold = median(best_thresholds)
    n_estimator = median(best_n_estimators)
    max_depth = median(best_max_depth)

    #retrain to get the best model
    forest = RandomForestClassifier(n_estimators=int(n_estimator), max_depth=int(max_depth))
    forest.fit(best_f1_sample, best_f1_sample_result)

    file_name = 'trained_models/' + project_name + '_' + str(ver) + '_best_model.pkl'
    dump_file = open(file_name, 'wb')
    pickle.dump(forest, dump_file)
    pickle.dump(threshold, dump_file)
    pickle.dump(n_estimator, dump_file)
    pickle.dump(max_depth, dump_file)
                
    #grid_search.fit(X_train, Y_train)
    #return grid_search
    return X_test