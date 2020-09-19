from sklearn.model_selection import KFold
import numpy as np
X = np.array([[1, 2], [3, 4],[5,6],[7, 8]])
y = np.array([1, 0, 1, 1])
#表示划分成几等份
kf = KFold(n_splits=2)
#kf.split(x) 返回训练集和测试集的索引
for train_index, test_index in kf.split(X):
      print("Train:", train_index, "Validation:",test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]