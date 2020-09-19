from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
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



# 标准的 F1
print(f1_score(target_test, model.predict(data_test), average='weighted'))

#  Fβ
print(fbeta_score(target_test, model.predict(data_test), beta=1, average='weighted'))
# beta = 1 时，Fb 退化为标准 F1

print(fbeta_score(target_test, model.predict(data_test), beta=2, average='weighted'))
#查全率有更大影响

print(fbeta_score(target_test, model.predict(data_test), beta=0.5, average='weighted'))
#查准率有更大影响

