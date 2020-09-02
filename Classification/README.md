# Classification
## Task Description: Binary Classification
* 實作一個線性二元分類器，來根據人們的個人資料，判斷其年收入是否高於 50,000 美元
* X_train, Y_train, X_test
  * 非連續性資料: 例如教育程度、婚姻狀態...
  * 連續性資料: 例如年齡、資本損失...
  * X_train, X_test : 每一列包含一個510維的資料作為一個樣本
  * Y_train: label = 0 代表  “<= 50K” 、 label = 1 代表  “ >50K ”
## Implementation
### Logistic Regression
### 實作步驟: <br>
<img src="images/logistic.png" width=800 height=600 /> <br>
### Optimizer Selection: <br>
### Regularization: <br>
<img src="images/regularization.png" width=800 height=450 /> <br>
在加入regularization後，經過測試發現會比沒加入regularization來得更差，原因是regularization的用意是在防止分類器太貼合training data，也就是說分類的切面太崎嶇，但這次的線性分類器不會有這個問題，所以基本上不需要做regularization

---
### Porbabilistic Generative Model
### 實作步驟: <br>
<img src="images/generative.png" width=800 height=600 /> 
<img src="images/probability.png" width=800 height=600 /> 
