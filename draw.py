import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt("1.txt")
b = np.loadtxt("2.txt")
c = np.loadtxt("5.txt")
d = np.loadtxt("6.txt")

x = range(len(a))


plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.plot(x, a, label='total update')
plt.plot(x, b, label='greedy update')
#plt.plot(x, c, label='epsilon-greedy update')
#plt.plot(x, d, label='UCB update')
plt.legend(loc='lower right')


plt.show()
