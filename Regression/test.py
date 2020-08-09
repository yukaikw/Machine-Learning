#! /usr/bin/env python3
import sys, csv, os
import pandas as pd
import numpy as np

#testing
testdata = pd.read_csv('./data/test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x1 = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x1[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
    
mean_x1 = np.load('mean1.npy')
std_x1 = np.load('std1.npy')
mean_x2 = np.load('mean2.npy')
std_x2 = np.load('std2.npy')

test_x2 = np.empty([240, 18*9], dtype = float)
for i in range(len(test_x2)): 
    for j in range(len(test_x2[0])): 
        test_x2[i][j] = test_x1[i][j] ** 2

for i in range(len(test_x1)):
    for j in range(len(test_x1[0])):
        if std_x1[j] != 0:
            test_x1[i][j] = (test_x1[i][j] - mean_x1[j]) / std_x1[j]
for i in range(len(test_x2)):
    for j in range(len(test_x2[0])):
        if std_x2[j] != 0:
            test_x2[i][j] = (test_x2[i][j] - mean_x2[j]) / std_x2[j]

w1 = np.load('weight1.npy')
w2 = np.load('weight2.npy')
test_x1 = np.concatenate((np.ones([240, 1]), test_x1), axis = 1).astype(float)
test_x2 = np.concatenate((np.ones([240, 1]), test_x2), axis = 1).astype(float)
ans_y = np.dot(test_x1, w1) + np.dot(test_x2, w2)

#save prediction
import csv
with open('./result/predict.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)