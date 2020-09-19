from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

iris = load_iris()
clf = SVC(kernel='linear', C=1, random_state=0)
predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)

print(predicted)
print(metrics.accuracy_score(predicted, iris.target))