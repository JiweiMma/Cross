
from pylab import *
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target
#cross_val_predict返回和`y`相同尺寸的数组
#每一个entry是通过交叉验证的相应预测
predicted = cross_val_predict(lr,boston.data,y,cv=10)


#设置中文字体
myfont = matplotlib.font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc")
mpl.rcParams['axes.unicode_minus'] = False
#绘制
plt.scatter(y,predicted)
plt.plot([y.min(),y.max()],[y.min(),y.max()],"k--",lw=4)
plt.title(u'绘制交叉验证预测',fontproperties=myfont)
plt.xlabel(u'测度',fontproperties=myfont)
plt.ylabel(u'预测',fontproperties=myfont)
#显示绘制结果
plt.show()