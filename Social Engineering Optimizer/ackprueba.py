import numpy as np
import matplotlib.pyplot as plt

N=30
x = np.arange(-30,30+1/N,1/N)
print(x.shape)
# y = 20*np.exp(-0.2*np.sqrt(0.5*x**2))-np.exp(0.5*(np.cos(2*np.pi*x)))+np.exp(1)+20
y = 20*(1-np.exp(-0.2*np.sqrt(np.square(x))))+np.exp(1)-np.exp(np.cos(2*np.pi*x));
print(y.shape)
plt.plot(x,y)
plt.show()
