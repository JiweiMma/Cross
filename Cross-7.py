from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.datasets import load_iris
iris = load_iris()

scoring = ['precision_macro', 'recall_macro']
clf = SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,cv=5, return_train_score=False)

print(scores.keys())
print(scores['test_recall_macro'])