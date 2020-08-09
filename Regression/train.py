#! /usr/bin/env python3
import sys, csv, os
import pandas as pd
import numpy as np

#load
data = pd.read_csv('./data/train.csv', encoding = 'big5')

#pre-processing (remove unimportant features)
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()
feature = [1, 2, 3, 4, 10, 12, 13, 16, 17]
for i in range(240):
    for j in range(len(feature)):
        data.iloc[feature[j], :] = 0
        feature[j] += 18


#extract
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

x1 = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x1[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value

x2 = np.empty([12 * 471, 18 * 9], dtype = float)
for i in range(len(x2)): #12 * 471
    for j in range(len(x2[0])): #18 * 9 
        x2[i][j] = x1[i][j] ** 2

#normalize
mean_x1 = np.mean(x1, axis = 0) #18 * 9 
np.save('mean1.npy', mean_x1)
std_x1 = np.std(x1, axis = 0) #18 * 9 
np.save('std1.npy', std_x1)

mean_x2 = np.mean(x2, axis = 0) #18 * 9 
np.save('mean2.npy', mean_x2)
std_x2 = np.std(x2, axis = 0) #18 * 9 
np.save('std2.npy', std_x2)

for i in range(len(x1)): #12 * 471
    for j in range(len(x1[0])): #18 * 9 
        if std_x1[j] != 0:
            x1[i][j] = (x1[i][j] - mean_x1[j]) / std_x1[j]

for i in range(len(x2)): #12 * 471
    for j in range(len(x2[0])): #18 * 9 
        if std_x2[j] != 0:
            x2[i][j] = (x2[i][j] - mean_x2[j]) / std_x2[j]

#training
dim = 18 * 9 + 1
w1 = np.zeros([dim, 1])
w2 = np.zeros([dim, 1])
x1 = np.concatenate((np.ones([12 * 471, 1]), x1), axis = 1).astype(float)
x2 = np.concatenate((np.ones([12 * 471, 1]), x2), axis = 1).astype(float)

learning_rate = 1
iter_time = 50000
adagrad1 = np.zeros([dim, 1])
adagrad2 = np.zeros([dim, 1])
eps = 0.00000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x1, w1) + np.dot(x2, w2) - y, 2))/471/12) #rmse
    if(t%100==0):
        print(str(t) + ":" + str(loss))
    gradient1 = 2 * (np.dot(x1.transpose(), np.dot(x1, w1) + np.dot(x2, w2) - y)) #dim*1
    adagrad1 += gradient1 ** 2
    w1 = w1 - learning_rate * gradient1 / np.sqrt(adagrad1 + eps)

    gradient2 = 2 * (np.dot(x2.transpose(), np.dot(x1, w1) + np.dot(x2, w2) - y)) #dim*1
    adagrad2 += gradient2 ** 2
    w2 = w2 - learning_rate * gradient2 / np.sqrt(adagrad2 + eps)

np.save('weight1.npy', w1)
np.save('weight2.npy', w2)