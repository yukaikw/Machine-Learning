# Recurrent Neural Network
## Task Description: Text Sentiment Classification
* 本次task的data為twitter上收集到的推文，每則推文都會被標注為正面或負面，如:
 <img src="images/data_1.png" width=1300 height=20 /> <br>
 1：正面 <br>
<img src="images/data_0.png" width=1300 height=20 /> <br>
0：負面 <br>
* dataset分為training和testing，其中training dataset又分為labeled training data和unlabeled training data
  * labeled training data    : 20萬
  * unlabeled training data  : 120萬
  * testing data             : 20萬
* 希望利用training dataset訓練一個RNN model，來預測每個句子所帶有的情緒
