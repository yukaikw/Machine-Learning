import os, sys
# 讀取 label.csv
import pandas as pd
# 讀取圖片
from PIL import Image
import numpy as np

import torch
# Loss function
import torch.nn.functional as F
from torch.autograd import Variable
# 讀取資料
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
# 載入預訓練的模型
import torchvision.models as models
# 將資料轉換成符合預訓練模型的形式
import torchvision.transforms as transforms
# 顯示圖片
import matplotlib.pyplot as plt

device = torch.device("cuda")

# 實作一個繼承 torch.utils.data.Dataset 的 Class 來讀取圖片
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 由 main function 傳入的 label
        self.label = torch.from_numpy(label).long()
        # 由 Attacker 傳入的 transforms 將輸入的圖片轉換成符合預訓練模型的形式
        self.transforms = transforms
        # 圖片檔案名稱的 list
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 圖片相對應的 label
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        # 由於已知這次的資料總共有 200 張圖片 所以回傳 200
        return 200

class Attacker:
    def __init__(self, img_dir, label):
        # 讀入預訓練模型 vgg16
        self.model = models.densenet121(pretrained = True)
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        # 利用 Adverdataset 這個 class 讀取資料
        self.dataset = Adverdataset(sys.argv[1]+'/images', label, transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)

    # FGSM 攻擊
    def fgsm_attack(self, image, epsilon, target, iteration):
        # 找出 gradient 的方向
        alpha = epsilon/iteration
        g = 0
        x = image
        for i in range(iteration):
            # 將圖片加上 gradient 方向乘上 epsilon 的 noise
            x = Variable(x.data, requires_grad=True)
            output = self.model(x)
            loss = F.cross_entropy(output, target)
            self.model.zero_grad()
            loss.backward()
            grad = x.grad.data
            g = 0.1 * g + grad/(grad.abs().sum())
            x = x + alpha * g.sign()
        return x
    
    def attack(self, epsilon, iteration):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_examples = []
        wrong, fail, success = 0, 0, 0
        f = []
        cnt = 0
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = data;
            data.requires_grad = True
            # 將圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]
            # 如果 class 錯誤 就不進行攻擊
            
            if init_pred.item() != target.item():
                data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                adv_examples.append( data_raw )
                wrong += 1
                cnt += 1
                continue
            # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
            
            
            perturbed_data = self.fgsm_attack(data, epsilon, target, iteration)
            
                
            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
            adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
            adv_examples.append( adv_ex )
            cnt += 1       
        final_acc = (success + wrong) / (success + fail + wrong)
        return adv_examples, final_acc


# 讀入圖片相對應的 label
df = pd.read_csv(sys.argv[1]+"/labels.csv")
df = df.loc[:, 'TrueLabel'].to_numpy()
label_name = pd.read_csv(sys.argv[1]+"/categories.csv")
label_name = label_name.loc[:, 'CategoryName'].to_numpy()
# new 一個 Attacker class
attacker = Attacker(sys.argv[1]+'/images', df)
# 要嘗試的 epsilon
epsilons = [0.01712]

accuracies, examples = [], []

# 進行攻擊 並存起正確率和攻擊成功的圖片
for eps in epsilons:
    ex, acc = attacker.attack(eps, 15)
    accuracies.append(acc)
    examples.append(ex)

transform = transforms.Compose([
	transforms.ToTensor(),
    transforms.ToPILImage(), 
    transforms.Resize(size=(224, 224)),                                   
])

for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        ex = examples[i][j]
        ex = np.transpose(ex, (1, 2, 0))
        ex = transform(ex)
        if j < 10:
        	filename = "00"+str(j)+".png"
       	elif j >= 10 and j < 100:
       		filename = "0"+str(j)+".png"
       	else:
       		filename = str(j)+".png"
       	ex.save(os.path.join(sys.argv[2], filename))
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
])

def calc(model, ori, adv):
    _, y1 = model(transform(ori).unsqueeze(0).to(device)).topk(1)
    _, y2 = model(transform(adv).unsqueeze(0).to(device)).topk(1)
    linf = np.linalg.norm((np.array(ori).astype('int64') - np.array(adv).astype('int64')).flatten(), np.inf)
    return linf
fnames = []
for i in range(200):
    fnames.append("{:03d}".format(i))
cnt = 0
for i in fnames:
    org = Image.open(os.path.join("./data/images", i + '.png'))
    adv = Image.open(os.path.join("./output", i + '.png'))
    cnt += calc(models.densenet121(pretrained = True).cuda(), org, adv)
print("linf = ", cnt/200)
"""