# semi-supervised.py
# 這個 block 是用來做self-training的
import os
import pickle
import torch
import argparse
import numpy as np
import pandas as pd
from utils import load_training_data, load_testing_data, evaluation
from w2v import train_word2vec
from preprocess import Preprocess
from data import TwitterDataset
from model import LSTM_Net
from train import training
from test import testing
from torch import nn
from gensim.models import word2vec

path_prefix = './'
# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 處理好各個 data 的路徑
train_with_label = os.path.join(path_prefix, 'training_label.txt')
train_no_label = os.path.join(path_prefix, 'training_nolabel.txt')
w2v_path = os.path.join(path_prefix, 'w2v_all.model') # 處理 word to vec model 的路徑

# 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
sen_len = 35
fix_embedding = True # fix embedding during training
bidirectional = True
batch_size = 16
model_num = 1
epoch = 5
lr = 0.001

model_dir = './model' # model directory for checkpoint model


# semi-supervised
train_x_no_label = load_training_data(train_no_label)
preprocess = Preprocess(train_x_no_label, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x_no_label = preprocess.sentence_word2idx()
train_dataset = TwitterDataset(X=train_x_no_label, y=None)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)
outputs = []
for i in range(model_num):
	model = torch.load(os.path.join(model_dir, 'ckpt'+str(i+1)+'.model'))
	outputs.append(testing(batch_size, train_loader, model, device))

# soft-voting ensemble
results = []
for j in range(len(outputs[0])):
	avg = 0
	for i in range(model_num):
		avg += outputs[i][j]
	avg /= model_num
	results.append(avg)

print("loading data ...")
train_x, y = load_training_data(train_with_label)
train_x_no_label = load_training_data(train_no_label)

# hard pseudo labeling
for i in range(len(results)):
	if results[i] >= 0.9:
		train_x.append(train_x_no_label[i])
		y.append(1)
	if results[i] <= 0.1:
		train_x.append(train_x_no_label[i])
		y.append(0)

print("saving results into files ... ")

with open('train_x', 'wb') as fp:
    pickle.dump(train_x, fp)

with open('y', 'wb') as fp:
    pickle.dump(y, fp)
