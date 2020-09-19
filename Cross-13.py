import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# 导入数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
#去掉了label为2，label只能二分，才可以。
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape

# 添加噪声特征
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

#分类，做ROC分析
#使用6折交叉验证，并且画ROC曲线
cv = StratifiedKFold(n_splits=6)

# 注意这里的应该改为probability=True以概率形式输出
#注意这里，probability=True,需要，不然预测的时候会出现异常。另外rbf核效果更好些
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
# k折交叉验证
for (train, test), color in zip(cv.split(X, y), colors):
    # 通过训练数据，建立模型，并对测试集进行测试，求出预测得分
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    #　注意这里返回的阈值，以区分正负样本的阈值
    # 通过roc_curve()函数，求出fpr和tpr，以及阈值
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])

    # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    mean_tpr += interp(mean_fpr, fpr, tpr)
    # 初始处为0
    mean_tpr[0] = 0.0
    #计算auc的值
    roc_auc = auc(fpr, tpr)
    # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i += 1
# 画对角线
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

#在mean_fpr，每个点处插值插值多次取平均
mean_tpr /= cv.get_n_splits(X, y)
# 坐标最后一个点为（1,1）
mean_tpr[-1] = 1.0
#计算平均AUC值
mean_auc = auc(mean_fpr, mean_tpr)
#画平均ROC曲线
#print mean_fpr,len(mean_fpr)
#print mean_tpr
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()