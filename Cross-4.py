from sklearn.model_selection import LeaveOneOut
import numpy as np
X = np.array([[1, 2], [3, 4],[5,6],[7, 8]])
y = np.array([1, 2, 2, 1])
loo = LeaveOneOut()
loo.get_n_splits(X)
for train_index, test_index in loo.split(X):
        print("train:", train_index, "validation:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]