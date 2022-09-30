import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt("offline_3.txt")
b = np.loadtxt("online_3.txt")
c = np.loadtxt("ground_truth_3.txt")

n = len(c)
x = np.asarray(range(n))
'''
plt.xlabel('m/n')
plt.ylabel('average bias')

temp1 = a[:, 0]
temp2 = b[:, 0]
temp1 = np.abs(temp1 - c)
temp2 = np.abs(temp2 - c)
for i in range(n):
    temp1[i] /= 5 * (i + 1)
    temp2[i] /= 5 * (i + 1)
plt.plot(x, temp1[0:n], color='green', label='abtests')
plt.plot(x, temp2[0:n], color='red', label='optimal experiment')

plt.legend(loc='upper right')

plt.grid()
plt.show()
'''
plt.xlabel('m/n')
plt.ylabel('variance')

temp1 = a[:, 1]
temp2 = b[:, 1]
plt.plot(x, temp1[0:n], color='green', label='offline')
plt.plot(x, temp2[0:n], color='red', label='online')

plt.legend(loc='lower right')

plt.grid()
plt.show()