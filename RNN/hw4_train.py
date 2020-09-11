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
from sklearn.model_selection import train_test_split

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
batch_size = 32
epoch = 5
lr = 0.001

model_dir = './model' # model directory for checkpoint model

"""
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
for i in range(8):
	model = torch.load(os.path.join(model_dir, 'ckpt'+str(i+1)+'.model'))
	outputs.append(testing(batch_size, train_loader, model, device))

# soft-voting ensemble
results = []
for j in range(len(outputs[0])):
	avg = 0
	for i in range(8):
		avg += outputs[i][j]
	avg /= 8
	results.append(avg)

print("loading data ...")
train_x, y = load_training_data(train_with_label)
train_x_no_label = load_training_data(train_no_label)
for i in range(len(results)):
	if results[i] >= 0.9:
		train_x.append(train_x_no_label[i])
		y.append(1)
	if results[i] <= 0.1:
		train_x.append(train_x_no_label[i])
		y.append(0)
	print("finish")

print("writing result to file")

with open('train_x', 'wb') as fp:
    pickle.dump(train_x, fp)

with open('y', 'wb') as fp:
    pickle.dump(y, fp)
"""
with open ('train_x', 'rb') as fp:
    train_x = pickle.load(fp)
with open ('y', 'rb') as fp:
    y = pickle.load(fp)

# 對 input 跟 labels 做預處理
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

# 製作一個 model 的對象
model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=3, dropout=0.5, fix_embedding=fix_embedding, bidirectional=bidirectional)
model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

# 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
X_train, X_val, y_train, y_val = train_x[100000:700000], train_x[:100000], y[100000:700000], y[:100000]

# 把 data 做成 dataset 供 dataloader 取用
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

# 把 data 轉成 batch of tensors
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8)

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)

# 開始訓練
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)