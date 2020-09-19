import numpy as np
import matplotlib.pyplot as plt

x = 2.5
xp = [1, 2, 3]
fp = [3, 2, 0]
y = np.interp(x, xp, fp)  # 1.0
plt.plot(xp, fp, '-o')
plt.plot(x, y, 'x')
plt.show()
