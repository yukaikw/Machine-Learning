# Classification
## Task Description: Binary Classification
* 實作一個線性二元分類器，來根據人們的個人資料，判斷其年收入是否高於 50,000 美元
* X_train, Y_train, X_test
  * discrete features in train.csv => one-hot encoding in X_train (education, martial state...)
  * continuous features in train.csv => remain the same in X_train (age, capital losses...).
  * X_train, X_test : each row contains one 510-dim feature represents a sample.
  * Y_train: label = 0 means  “<= 50K” 、 label = 1 means  “ >50K ”
