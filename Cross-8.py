from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.metrics import make_scorer,recall_score
from sklearn.datasets import load_iris
iris = load_iris()

scoring = {'prec_macro': 'precision_macro','rec_micro': make_scorer(recall_score, average='macro')}

clf = SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,cv=5, return_train_score=False)

print(scores.keys())
print(scores['test_rec_micro'])