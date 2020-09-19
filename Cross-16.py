from pandas import np
import numpy as np
import matplotlib.pyplot as plt
x = [0, 1, 1.5, 2.72, 3.14]
xp = [1, 2, 3]
fp = [3, 2, 0]
y = np.interp(x, xp, fp)  # array([ 3. ,  3. ,  2.5 ,  0.56,  0. ])
plt.plot(xp, fp, '-o')
plt.plot(x, y, 'x')
plt.show()
