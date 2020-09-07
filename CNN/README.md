# Convolutional Neural Network
## Task Description: Image Classification
* 此次資料集為網路上蒐集到的食物照片，共有11類
( Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit )
  * Training set: 9866張
  * Validation set: 3430張
  * Testing set: 3347張
* 利用資料集來訓練出一個CNN model，以此預測出每張圖片的食物種類
<img src="images/testingdata.png" width=900 height=200 /> <br>
## Download Dataset
<img src="images/dataset.png" width=700 height=60 /> <br>
## Implementation
### Model Selection
在CNN模型的架構上，我選擇的是VGG16，如下圖所示:
<img src="images/vgg16.png" width=800 height=300 /> <br>
其中包括13層Convolution Layer，5層MaxPooling，3層Fully-Connected Layer，而會選擇vgg16的原因是深度較淺的CNN模型經過測試後表現較差，但是也並非深度越深效果越好，深度越深的模型在training時花費的時間越多，也可能會因為資料量不足而導致overfitting的情況更加嚴重，而詳細的模型架構和參數量如下圖(torchsummary)所示: <br>
<img src="images/torchsummary.png" width=450 height=900 /> <br>
* Conv2d()
* BatchNorm2d()
* ReLU()
### Optimizer Selection
在Optimizer的選擇上，我選擇使用Adam + SGDM，其中選擇Adam的優點在於收斂的速度較快，可以節省許多training花費的時間，但缺點是收斂的結果較差，因此我選擇在前200個epoch使用Adam，而後50個epoch使用收斂結果較好的SGDM，兩者結合後在testing data上得到的準確率會比單獨使用Adam來的高
### Data Augmentation
### Confusion Matrix
---
### Reference:
投影片部份取自李宏毅教授的機器學習課程 (
[CNN](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/CNN.pdf)
[作業說明投影片](https://docs.google.com/presentation/d/1_6TJrFs3JGBsJpdRGLK1Fy_EiJlNvLm_lTZ9sjLsaKE/edit#slide=id.p4)
[kaggle連結](https://reurl.cc/ZO7XpM) )
