#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019. 
Deep Anomaly Detection with Deviation Networks. 
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""

import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.externals.joblib import Memory
from sklearn.metrics import average_precision_score, roc_auc_score

mem = Memory("./dataset/svm_data")


@mem.cache
def get_data_from_svmlight_file(path):
    data = load_svmlight_file(path)
    return data[0], data[1]


def dataLoading(path):
    # loading data
    #CHANGED MADE HERE ALMOST WHOLE THING CHANGED
    
    #ORIGINAL FUNCTION
    df = pd.read_csv(path)

    labels = df['class']

    x_df = df.drop(['class'], axis=1)

    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)

    return x, labels
    '''
    df = pd.read_csv(path)

    labels = df['Value'] #CHANGED MADE HERE

    #x_df = df.drop(['Date'], axis=1)

    x = labels #CHANGED MADE HERE
    #print("Data shape: (%d, %d)" % x.shape)

    return x, labels
   '''

def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap


def writeResults(name, n_samples, dim, n_samples_trn, n_outliers_trn, n_outliers, depth, rauc, ap, std_auc, std_ap,
                 train_time, test_time, path="./results/auc_performance_cl0.5.csv"):
    csv_file = open(path, 'a')
    row = name + "," + str(n_samples) + "," + str(dim) + ',' + str(n_samples_trn) + ',' + str(
        n_outliers_trn) + ',' + str(n_outliers) + ',' + str(depth) + "," + str(rauc) + "," + str(std_auc) + "," + str(
        ap) + "," + str(std_ap) + "," + str(train_time) + "," + str(test_time) + "\n"
    csv_file.write(row)

def writeperformance(score,anomaly,y_test,x_test, path="./results/result_performance_cl0.5.csv"):
    csv_file = open(path, 'a')
    row = str(score) + "," + str(anomaly) + "," + str(y_test) + ',' + str(x_test) +  "\n"
    csv_file.write(row)
