import numpy as np
from sklearn.model_selection import ShuffleSplit

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([1, 2, 3, 4, 5, 6])
rs = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
rs.get_n_splits(X)
print(rs)
for train_index, test_index in rs.split(X, y):
    print("Train Index:", train_index, ",Test Index:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print(X_train,X_test,y_train,y_test)
print("==============================")
rs = ShuffleSplit(n_splits=3, train_size=0.5, test_size=0.25, random_state=0)
rs.get_n_splits(X)
print(rs)
for train_index, test_index in rs.split(X, y):
    print("Train Index:", train_index, ",Test Index:", test_index)