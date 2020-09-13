import os, sys
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

testing_data = os.path.join('./', 'testing_data.txt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w2v_path = os.path.join('./', 'w2v_all.model')

sen_len = 35
batch_size = 32
model_num = 1

test_x = load_testing_data(testing_data)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)
model_dir = './model'
outputs = []
for i in range(model_num):
	model = torch.load(os.path.join(model_dir, 'ckpt'+str(i+1)+'.model'))
	outputs.append(testing(batch_size, test_loader, model, device))

# soft-voting ensemble
results = []
for j in range(len(outputs[0])):
	avg = 0
	for i in range(model_num):
		avg += outputs[i][j]
	avg /= model_num
	if avg >= 0.5:
		results.append(1)
	else:
		results.append(0)

# 寫到 csv 檔案供上傳 Kaggle
tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":results})
tmp.to_csv(os.path.join('./result/predict.csv'), index=False)
