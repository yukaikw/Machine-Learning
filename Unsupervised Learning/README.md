# Unsupervised Learning
## Task Description
* 分辨給定的兩張 images 是否為風景 (植物也算風景 e.g., 一片葉子)
* Dataset:
  * trainX.npy
    * 裡面總共有 8500 張 RGB 圖片，大小都是 32 * 32 * 3
    * shape 為 (8500, 32, 32, 3)
  * valX.npy
    * 裡面總共有 500 張 RGB 圖片，大小都是 32 * 32 * 3
    * shape 為 (500, 32, 32, 3)
  * valY.npy
    * 對應 valX.npy 的 label
    * shape為 (500,)

## Download Dataset
## Implementation

---
### Reference:
投影片部份取自李宏毅教授的機器學習課程 (
[Unsupervised Learning](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/PCA%20(v3).pdf)
[作業說明投影片](https://docs.google.com/presentation/d/1ULbTKqn7ikFOTU-r0DoqAca6lej3QmLWwORfcr-0F3o/edit#slide=id.g7be340f71d_0_0) )
