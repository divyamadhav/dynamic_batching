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
from sklearn.model_selection import KFold
from matplotlib import pyplot
from statistics import median
import pickle
import csv
import warnings
import datetime
import multiprocess
warnings.filterwarnings("ignore")

import sys
sys.path.append("../")
from projects import project_list

def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:  
        median = (data[size // 2] + data[size // 2 - 1]) / 2
        data[0] = median
    if size % 2 == 1:  
        median = data[(size - 1) // 2]
        data[0] = median
    return data[0]

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

def output_values(Y_data):
    Y_t = []
    for e in Y_data:
        if e == 'passed':
            Y_t.append(1)
        else:
            Y_t.append(0) 
    return Y_t

def pd_get_train_test_data(file_path, first_failures=True):
    columns = ['tr_build_id', 'git_num_all_built_commits', 'git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_modified', 'tr_status']
    X = pd.read_csv(file_path, usecols = columns)
    X['tr_status'] = output_values(X['tr_status'])
    
    if first_failures:
        X = get_first_failures(X)

    return X


def sbs_train(path, ver, train=1):

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



def sbs_test(p, ver):
    
    
    global batch_total
    global batch_durations
    
    p_name = p.split('/')[1]

    
    X_test = sbs_train(p, ver, train=0)
    if len(X_test) == 0:
        return
    
    dir_name = p_name[:-4].split('-', 1)[1]
    model_file_name = '../../Models/' + dir_name + '/' + p_name + '_' + str(ver) + '_best_model.pkl'
    model_file = open(model_file_name, 'rb')
    predictor = pickle.load(model_file)
    threshold = pickle.load(model_file)
    
    Y_test = X_test['tr_status'].tolist()

    X_test.drop('tr_build_id', inplace=True, axis=1)
    X_test.drop('tr_status', inplace=True, axis=1)


    y_pred = predictor.predict(X_test)
    print(y_pred)
    print(Y_test)

    print(precision_score(Y_test, y_pred))
    print(recall_score(Y_test, y_pred))
    
    result_df = pd.DataFrame()
    #Since we have already divided train and test data, we don't need to collect build ids again
    result_df['Build_Result'] = y_pred
    result_df['Actual_Result'] = Y_test
    result_df['Index'] = list(range(1, len(Y_test)+1))

    #print(commit_values)
    headers = ['Build_Result', 'Actual_Result']

    file_name = './final_sbs_results/' + p_name.split('.')[0] + '_' + str(ver) + '_200_metrics.csv'
    result_df.to_csv(file_name)



# for pr in project_list:
#     for i in range(0,10):
#         sbs_test(pr, i)











def validation():
    global project_list

    batch_result = 'final_sbs_result.csv'
    final_file = open(batch_result, 'w')
    final_headers = ['project', 'method', 'reqd_builds', 'delay']
    final_writer = csv.writer(final_file)
    final_writer.writerow(final_headers)

    for proj in project_list:
            project = proj.split('/')[1]


            #file to write results of SBS algorithm
            proj_result = 'results/' + project.split('.')[0] + '_result.csv'
            result_file = open(proj_result, 'w')
            res_headers = ['index', 'duration', 'total_builds']
            res_writer = csv.writer(result_file)
            res_writer.writerow(res_headers)
            
            csv_file = pd.DataFrame()
            
            for v in range(0, 10):
                file_name = './final_sbs_results/' + project.split('.')[0] + '_' + str(v) + '_200_metrics.csv'
                ver_results = pd.read_csv(file_name)
                csv_file = csv_file.append(ver_results)
                

            actual_results = csv_file['Actual_Result'].tolist()
            pred_results = csv_file['Build_Result'].tolist()

            delay_indexes = []
            built_indexes = []
            first_failure = 0
            ci = []

            total_builds = len(actual_results)
            sbs_builds = 0

            for i in range(len(actual_results)):

                #If first failure is already found, continue building until actual build pass is seen
                if first_failure == 1:
                    ci.append(0)
                    sbs_builds += 1

                    if actual_results[i] == 1:
                        #actual build pass is seen, switch to prediction
                        first_failure = 0
                    else:
                        first_failure = 1
                else:
                    #we're in prediction state, if predicted to skip, we skip
                    if pred_results[i] == 1:
                        ci.append(1)
                    else:
                        #if predicted to fail, we switch to determine state and set first_failure to True
                        ci.append(0)
                        sbs_builds += 1
                        first_failure = 1-actual_results[i]


            total_builds = len(ci)
            actual_builds = ci.count(0)

            saved_builds = 100*ci.count(1)/total_builds
            reqd_builds = 100*ci.count(0)/total_builds

            for i in range(len(ci)):
                if ci[i] == 0:
                    built_indexes.append(i)
                else:
                    if actual_results[i] == 0:
                        delay_indexes.append(i)


            '''from_value = 0
            delay = []
            for k in range(len(built_indexes)):
                for j in range(len(delay_indexes)):
                    if delay_indexes[j] > from_value and delay_indexes[j] < built_indexes[k]:
                        delay.append(built_indexes[k] - delay_indexes[j])
                from_value = built_indexes[k]

            final_index = len(ci)

            for j in range(len(delay_indexes)):
                if delay_indexes[j] > from_value and delay_indexes[j] < final_index:
                    delay.append(final_index - delay_indexes[j])'''

            bp = 0
            mp = 0
            temp_delay = 0
            total_delay = []

            while bp < len(built_indexes):
                while mp < len(delay_indexes) and delay_indexes[mp] < built_indexes[bp]:
                    temp_delay = built_indexes[bp] - delay_indexes[mp]
                    print("Difference: {}, Built_index = {} , Missed_index = {}".format(temp_delay, built_indexes[bp], delay_indexes[mp]))
                    total_delay.append(temp_delay)
                    mp += 1
                bp += 1

            while mp < len(delay_indexes):
                temp_delay = total_builds - delay_indexes[mp]
                print("Difference: {}, Built_index = {} , Missed_index = {}".format(temp_delay, total_builds, delay_indexes[mp]))
                total_delay.append(temp_delay)
                mp += 1


            delay = total_delay
            print(total_delay)
            if len(total_delay) == 0:
                total_delay = [0]

            print('saved_builds for {} is {}'.format(project, saved_builds))
            print('delay for {} is {}\n\n'.format(project, sum(delay)))
            final_writer.writerow([project, 'sbs', reqd_builds, median(total_delay)])


            
            batch_builds = 0
            commit_num = 1
            build_time = 0

#             for i in range(len(ci)):

#                 if commit_num == batch_size:
#                     res_writer.writerow([i+1, build_time, batch_builds])
#                     commit_num = 1
#                     build_time = 0
#                     batch_builds = 0
#                     continue

#                 if ci[i] == 0:
#                     batch_builds += 1
                    

#                 commit_num += 1


            # file_name = 'metrics/' + project.split('.')[0] + '_real_metrics.csv'

            # csv_file = csv.reader(open(file_name, 'r'))

            # built_commits = []
            # build_time = 0
            # total_builds = 0
            # actual_builds = 0
            # commit_num = 1
            # flag = 0
            # batches = []
            # num = 0
            # b_id = 0





            # 	# if a build is predicted to fail, they will build it
            # 	if build[-2] == '0':
            # 		#add the build time
            # 		build_time += int(build[2])
            # 		actual_builds += 1
            # 		total_builds += 1
            # 		b_id = build[0]
            # 		flag = 1

            # 	#if prev build has failed, build until you see a true build pass
            # 	if flag == 1:
            # 		if build[-1] == '0':
            # 			if b_id != build[0]:
            # 				build_time += int(build[2])
            # 				actual_builds += 1
            # 				total_builds += 1				
            # 		if build[-1] == '1':
            # 			#this is the first build pass after failure
            # 			#go back to predicting
            # 			if b_id != build[0]:
            # 				build_time += int(build[2])
            # 				actual_builds += 1
            # 				total_builds += 1
            # 			flag = 0


            # 	'''#if a build passes,
            # 	if build[-2] == '1':
            # 		#check if this is the first build pass after failure
            # 		if (flag == 1):
            # 			flag = 0
            # 			build_time += int(build[2])
            # 			total_builds += 1'''

            # 	if commit_num == 4:
            # 		batches.append([int(build[1]), build_time, total_builds])
            # 		res_writer.writerow([int(build[1]), build_time, total_builds])
            # 		commit_num = 0
            # 		built_commits.append(build_time)
            # 		build_time = 0
            # 		total_builds = 0

            # 	commit_num += 1
            # #print(batches)
            # #print(total_builds)
            # print(actual_builds)
            # #print(len(csv_file))
            # #print('Total time taken for builds:')
            # #print(built_commits)

validation()
