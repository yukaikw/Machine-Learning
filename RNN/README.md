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
由於1-of-N encoding和BOW這兩種方法都有明顯的缺點，因此我最終選擇prediction-based embedding中的skip-gram，其中skip-gram和CBOW都可以直接使用gensim這個套件來完成，不需要親自手刻
### Preprocessing
* pad sequence: 將每個句子變成一樣的長度，經過測試後發現當長度為35時有最佳的結果，如果句子長度過短的話，可能會有沒將重要資訊讀入的情況
* remove stopword: 移除資料中出現特別頻繁的詞，如"a", "the", "is", "are"，但是經過測試後發現反而會使準確率下降，因此最後沒有使用
* remove number & symbol: 移除資料中出現的數字和符號，經過測試後也同樣發現會使準確率下降，因此最後沒有使用
### Model Selection
<img src="images/lstm.png" width=700 height=500 /> <br>
<br>
在模型的選擇上，我使用的是LSTM，比起較為簡易的GRU，LSTM有著更好的準確率，而參數部份input_size = 250，hidden_size = 150, num_layers = 3, batch_first = True, dropout = 0.5, bidirectional = True，bidirectional LSTM在準確率上會比unidirectional LSTM稍微好一點，而訓練時設定為fix embedding，也就是word embedding不會和模型一起訓練，如果將兩者一起訓練會使得overfit的情況更加嚴重
### BOW + DNN v.s. RNN
1. today is a good day, but it is hot <br>
2. today is hot, but it is a good day <br>

用以上兩個句子為例，在BOW + DNN模型上過softmax後的預測數值皆為70.9%，而在RNN上分別為26.3%和98.7%，由於BOW只會紀錄單字出現的次數，而不會去考慮順序，因此對兩者的預測分數相同，RNN則會考慮句子中的順序，並且利用forget gate來決定是否保留前面的單字，其中"good"為正面的單字，"hot"為較負面的單字，所以BOW會認為這兩個句子為中立偏正面，而LSTM可能因為看到"but"這個單字時選擇遺忘前面的部份，因此判斷第一句為負面第二句為正面，從結果上來觀察，RNN在判斷上有著較強的能力，而對於整個testing data的準確率上，也是RNN勝過BOW + DNN，所以在這次的task中使用RNN較為合適，BOW + DNN可能比較適合只需要判斷句子主題或者當資料量不足的task
### Ensemble
透過建立多個模型來解決單一預測問題，其原理是利用dataset訓練出多個分類器，各自獨立學習和做出預測，最後將這些預測結果結合成單一預測，因此會優於單一個分類器做出的預測，也就是三個臭皮匠勝過一個諸葛亮的概念，其中結合預測結果的方法又主要分為hard-voting ensemble和soft-voting ensemble:
* Hard-Voting Ensemble: 根據少數服從多數來決定預測結果
* Soft-Voting Ensemble: 將多個模型預測概率的數值取平均作為預測結果，如果是使用加權平均作為預測結果，則稱為weighted average ensemble

舉例來說:

Model | label = 1 | label = 0
------|-----------|-----------
A     |   90%     |   10%
B     |   40%     |   60%
C     |   30%     |   70%

如果是使用hard-voting的話，最終預測結果為label = 0，因為B和C認為是label = 0的機率較高，而如果是使用soft-voting的話，最終預測結果為label = 1，因為將三者的預測機率做平均後，認為是label = 1的機率較高

在這次的task中，我選擇使用的是soft-voting ensemble，比起hard-voting ensemble有著更好的準確率，而我總共訓練了8個模型做ensemble，其中有4個bidirectional LSTM和4個unidirectional LSTM，比起單一個LSTM模型，在testing data上的準確率可以從82.6%提升到83.2%
### Semi-Supervised Learning
把訓練好的模型對unlabeled data做預測，並將這些預測後的值轉成該筆unlabeled data的label，並加入這些新的data做訓練，以此增加labeled data的數量，其中標記label的方法又分為hard pseudo labeling和soft pseudo labeling:
* Hard Pseudo Labeling: 當預測的數值大於設定的boundary則將label標記為1，反之，當預測的數值小於設定的boundary則將label標記為0
* Soft Pseudo Labeling: 將預測的數值直接當作data的label
在這次的task中，
---
### Reference:
投影片部份取自李宏毅教授的機器學習課程 (
[RNN](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/RNN%20(v2).pdf)
[Word Embedding](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/word2vec%20(v2).pdf)
[Semi-Supervised](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/semi%20(v3).pdf)
[作業說明投影片](https://docs.google.com/presentation/d/1W5-D0hqchrkVgQxwNLBDlydamCHx5yetzmwbUiksBAA/edit#slide=id.g7cd4f194f5_2_151)
[kaggle連結](https://www.kaggle.com/c/ml2020spring-hw4) )
