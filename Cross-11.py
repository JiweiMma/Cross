import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Users\lenovo\PycharmProjects\untitled4\venv\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\maozedong.ttf", size=20)
sentences = [
    '沁园春·长沙',
    '现代 ***',
    '独立寒秋，湘江北去，橘子洲头。',
    '看万山红遍，层林尽染；',
    '漫江碧透，百舸争流。',
    '鹰击长空，鱼翔浅底，',
    '万类霜天竞自由。',
    '怅寥廓，问苍茫大地，谁主沉浮？',

    '携来百侣曾游，',
    '忆往昔峥嵘岁月稠。',
    '恰同学少年，风华正茂；',
    '书生意气，挥斥方遒。',
    '指点江山，激扬文字，',
    '粪土当年万户侯。',
    '曾记否，到中流击水，浪遏飞舟！'
]
plt.axis('off')
for i in range(len(sentences)):
    plt.text(0.2, 1 - 0.1 * i, sentences[i], fontproperties=font)  # 横纵最大都是1
plt.show()


