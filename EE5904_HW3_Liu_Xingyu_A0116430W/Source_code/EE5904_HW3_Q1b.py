#%% Q1b
import numpy as np
import matplotlib.pyplot as plt

x_train = np.arange(-1, 1.01, 0.05)
x_test = np.arange(-1, 1.01, 0.01)
y_train = 1.2*np.sin(np.pi*x_train)-np.cos(2.4*np.pi*x_train)+0.3*np.random.normal(0, 1, x_train.shape)
y_test = 1.2*np.sin(np.pi*x_test)-np.cos(2.4*np.pi*x_test)

index = np.random.choice(41,15)
centers = np.ones(15)
for i in range(15):
    centers[i] = x_train[index[i]]
    
dis = []
for i in range(15):
    for j in range(15):
        dis.append(np.abs(centers[i]-centers[j]))
dismax = np.max(dis)

inter_mat = np.zeros((41,15))
M = 15

for j in range(41):
    for i in range(15):
        inter_mat[j,i] = np.exp(-M*((x_train[j]-centers[i])**2)/(dismax**2))
weights = np.dot(np.linalg.pinv(inter_mat), y_train)

result_mat = np.zeros((201,15))
for j in range(201):
    for i in range(15):
        result_mat[j,i] = np.exp(-M*((x_test[j]-centers[i])**2)/(dismax**2))
y_pred = np.dot(result_mat, weights)

plt.plot(x_test, y_test, color = 'red', label = 'actual value')
plt.plot(x_test, y_pred, color = 'blue', label = 'prediction')
plt.xlabel('x value')
plt.ylabel('function value')
plt.title('Function Approximation')
plt.legend(loc = 'upper left')
plt.savefig('Q1b.png')
plt.show()
