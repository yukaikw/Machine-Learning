import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.autograd import Variable
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 32 * 32 * 3
            ), nn.Tanh())

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2, x1

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
    
print('load testing data ...')
test = np.load(sys.argv[1], allow_pickle = True)
test = test.reshape(len(test), -1)

outlier = 1
enhance = 3000
batch_size = 128
data = torch.tensor(test, dtype = torch.float)

model = fcn_autoencoder().cuda()
model.load_state_dict(torch.load(sys.argv[2]))

test_dataset = TensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler = test_sampler, batch_size = batch_size)
    
model.eval()
reconstructed = list()

model_path = sys.argv[2]
if model_path[-8:] == "best.pth":
    for data in test_dataloader: 
        img = data[0].cuda()
        output = model(img)
            
        reconstruct = output[0].cpu().detach().numpy()
        reconstruct = reconstruct.reshape(len(reconstruct), -1)
        reconstructed.append(reconstruct)

    reconstructed = np.concatenate(reconstructed, axis = 0)
    kmeans_x = MiniBatchKMeans(n_clusters = 5, random_state = 0).fit(reconstructed)

    cluster = kmeans_x.predict(reconstructed)
    anomality = np.sum(np.square(kmeans_x.cluster_centers_[cluster] - reconstructed), axis = 1)
    anomality[cluster == outlier] += enhance

else:
    for data in test_dataloader: 
        img = data[0].cuda()
        output = model(img)
        reconstructed.append(output[0].cpu().detach().numpy())

    reconstructed = np.concatenate(reconstructed, axis = 0)
    anomality = np.sqrt(np.sum(np.square(reconstructed - test).reshape(len(test), -1), axis = 1))

with open(sys.argv[3], 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(anomality)):
        f.write('{},{}\n'.format(i + 1, anomality[i]))
