# Recurrent Neural Network
## Task Description: Text Sentiment Classification
* 本次task的data為twitter上收集到的推文，每則推文都會被標注為正面或負面，如:
 <img src="images/data_1.png" width=1300 height=20 /> <br>
 1：正面 <br>
<img src="images/data_0.png" width=1300 height=20 /> <br>
0：負面 <br>
* dataset分為training和testing，其中training dataset又分為labeled training data和unlabeled training data
  * labeled training data    : 20萬 （句子配上0 or1，+++$+++ 只是分隔符號）
  * unlabeled training data  : 120萬 （只有句子，用來做semi-supervised learning)
  * testing data             : 20萬
* 希望利用training dataset訓練一個RNN model，來預測每個句子所帶有的情緒
## Download Dataset
<img src="images/dataset.png" width=700 height=60 /> <br>
## Implementation
### Word Embedding
Word embedding是一種將單字轉換為向量的方法，常見的word embedding方法有三類，大致上分為1-of-N Encoding，Bag-of-Words，和Prediction-Based Embedding: <br>
<img src="images/1-of-N.png" width=720 height=360 /> <br>
<br>
<img src="images/BOW.png" width=720 height=360 /> <br>
<br>
<img src="images/prediction-based.png" width=720 height=510 /> <br>
<br>
由於1-of-N encoding和BOW這兩種方法都有明顯的缺點，因此我最終選擇prediction-based embedding中的skip-gram，其中skip-gram和CBOW都可以直接使用gensim這個套件來完成，不需要親自實作
### Preprocessing
* pad sequence: 將每個句子變成一樣的長度，經過測試後發現當長度為35時有最佳的結果，如果句子長度過短的話，可能會有沒將重要資訊讀入的情況
* remove stopword: 移除資料中出現特別頻繁的詞，如"a", "the", "is", "are"，但是經過測試後發現反而會使準確率下降，因此最後沒有使用
* remove number & symbol: 移除資料中出現的數字和符號，經過測試後也同樣發現會使準確率下降，因此最後沒有使用
### Model Selection
### BOW + DNN v.s. RNN
### Ensemble
### Semi-Supervised Learning


---
### Reference:
投影片部份取自李宏毅教授的機器學習課程 (
[RNN](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/RNN%20(v2).pdf)
[作業說明投影片](https://docs.google.com/presentation/d/1W5-D0hqchrkVgQxwNLBDlydamCHx5yetzmwbUiksBAA/edit#slide=id.g7cd4f194f5_2_151)
[kaggle連結](https://www.kaggle.com/c/ml2020spring-hw4) )
