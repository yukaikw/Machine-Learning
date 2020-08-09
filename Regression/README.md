# Regression
## Task Description: PM2.5 Prediction
* 使用豐原站的觀測記錄分成, train set 跟 test set，train set 是豐原站每個月的前 20 天所有資料，test set 則是從豐原站剩下的資料中取樣出來
  * train.csv: 每個月前 20 天的完整資料
  * test.csv : 從剩下的資料當中取樣出連續的 10 小時為一筆，前九小時的所有觀測數據當作 feature，第十小時的 PM2.5 當作 answer，一共取出 240 筆不重複的 test data，請根據 feature 預測這 240 筆的 PM2.5
* Data 含有 18 項觀測數據 AMB_TEMP, CH4, CO, NHMC, NO, NO2, NOx, O3, PM10, PM2.5, RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR
## Implementation
實作linear regression的步驟:

<img src="graphic/step1.png" width=800 height=550 /> 
<img src="graphic/step2.png" width=800 height=550 /> 
<img src="graphic/step3.png" width=800 height=550 /> 

Model Selection: <br>
在進行model的選擇時, 我嘗試過以下三種model:  <br>

Optimizer Selection: <br>
Feature Selection: <br>
Regularization: <br>
<img src="graphic/regularization.png" width=800 height=550 /> 

Reference: 
[課程投影片](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/Regression.pdf)
[作業說明投影片](https://docs.google.com/presentation/d/18MG1wSTTx8AentGnMfIRUp8ipo8bLpgAj16bJoqW-b0/edit#slide=id.g4cd6560e29_0_15)
