from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# 引入数据集
dataset = load_iris()
data = dataset.data
target = dataset.target
features = dataset.feature_names

# 划分数据集以及模型训练
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.33, random_state=7)
model = DecisionTreeClassifier()
model.fit(data_train, target_train)


print(precision_score(target_test, model.predict(data_test), average='weighted'))

print(recall_score(target_test, model.predict(data_test), average='weighted'))
