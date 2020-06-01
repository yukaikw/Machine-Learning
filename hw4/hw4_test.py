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
from sklearn.model_selection import train_test_split

testing_data = os.path.join(sys.argv[1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w2v_path = os.path.join('./', 'w2v_all.model')

sen_len = 30
fix_embedding = True # fix embedding during training
batch_size = 128
epoch = 20
lr = 0.0005

test_x = load_testing_data(testing_data)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)
model_dir = './'
model = torch.load(os.path.join(model_dir, 'ckpt.model'))
outputs = testing(batch_size, test_loader, model, device)

# 寫到 csv 檔案供上傳 Kaggle
tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
tmp.to_csv(os.path.join(sys.argv[2]), index=False)
