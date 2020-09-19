import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Users\lenovo\PycharmProjects\untitled4\venv\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\maozedong.ttf", size=130)
plt.axis('off')
plt.text(0.01, 0.5, '马绩玮', fontproperties=font)
plt.show()
