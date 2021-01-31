import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.autograd import Variable
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


def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x : generating images
    x : origin images
    mu : latent mean
    logvar : latent log variance
    """
    mse = criterion(recon_x, x)
    # KL Divergence
    # loss = 0.5 * sum(1 + log(sigma ^ 2) - mu ^ 2 - sigma ^ 2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD


print('load training data ...')
train = np.load(sys.argv[1], allow_pickle = True)
train = train.reshape(len(train), -1)

nepoch = 1000
batch_size = 128
learning_rate = 1e-3
        
data = torch.tensor(train, dtype = torch.float)
train_dataset = TensorDataset(data)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler = train_sampler, batch_size = batch_size)

model = fcn_autoencoder().cuda()
    
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr = learning_rate)

model.train()
best_loss = np.inf
for epoch in range(nepoch):
    for data in train_dataloader:
        img = data[0].cuda()
        # =================== forward =====================
        output = model(img)
        loss = criterion(output[0], img)
        # =================== backward ====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # =================== save ========================
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), sys.argv[2])
    # =================== log =============================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, nepoch, loss.item()))