import os
import cv2
import itertools
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from model import Classifier

args = {
      'ckptpath': './vgg16.model',
      'dataset_dir': './food-11',
      'output_dir' : './output'
}
args = argparse.Namespace(**args)

model = torch.load(args.ckptpath)

def plotConfusionMatrix(confusionmatrix, listClasses):
    title = "ConfusionMatrix"
    cmap = plt.cm.jet

    confusionmatrix = confusionmatrix.astype("float") / confusionmatrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(confusionmatrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(listClasses))
    plt.xticks(tick_marks, listClasses, rotation=45)
    plt.yticks(tick_marks, listClasses)

    thresh = confusionmatrix.max() / 2.
    for i, j in itertools.product(range(confusionmatrix.shape[0]), range(confusionmatrix.shape[1])):
        plt.text(j, i, "{:.2f}".format(confusionmatrix[i, j]), horizontalalignment="center",
                color="white" if confusionmatrix[i, j] > thresh else "black")
    plt.tight_layout() # 自動調整間距
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(args.output_dir, 'confusionmatrix'))


def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = os.listdir(path)
    x = np.zeros((len(image_dir), 142, 142, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(142, 142))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

val_x, val_y = readfile(os.path.join(args.dataset_dir, "validation"), True)
val_set = ImgDataset(val_x, val_y, test_transform)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
prediction = []
label = []
with torch.no_grad():
    for i, data in enumerate(val_loader):
        val_pred = model(data[0].cuda())
        val_label = np.argmax(val_pred.cpu().data.numpy(), axis=1)
        for y in data[1].numpy():
          label.append(y)
        for y in val_label:
          prediction.append(y)

listClasses = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"]
arrayConfusionMatrix = confusion_matrix(label, prediction)
plotConfusionMatrix(arrayConfusionMatrix, listClasses=listClasses)