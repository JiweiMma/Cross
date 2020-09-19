from sklearn.model_selection import train_test_split
import numpy as np
X = np.array([[1, 2], [3, 4],[5,6],[7, 8]])
y = np.array([1, 2, 2, 1])
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.10, random_state = 5)
print("X_train:\n",X_train)
print("y_train:\n",y_train)
print("X_test:\n",X_test)
print("y_test:\n",y_test)