import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt("1.txt")

b = [abs(a[i+2]-a[0]) for i in range(len(a)-2)]
x = np.asarray(range(len(b)))/10


plt.xlabel('epsilon')
plt.ylabel('bias')

plt.plot(x, b, marker='s', color='green', label='our algorithm')
plt.axhline(y=abs(a[0]-a[1]), color='red', linestyle="--", label='A/B testing')
plt.legend(loc='upper right')

plt.grid()
plt.show()
