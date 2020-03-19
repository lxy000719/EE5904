#%% Q1a
import numpy as np
import matplotlib.pyplot as plt

x_train = np.arange(-1, 1.01, 0.05)
x_test = np.arange(-1, 1.01, 0.01)
y_train = 1.2*np.sin(np.pi*x_train)-np.cos(2.4*np.pi*x_train)+0.3*np.random.normal(0, 1, x_train.shape)
y_test = 1.2*np.sin(np.pi*x_test)-np.cos(2.4*np.pi*x_test)

inter_mat = np.zeros((41,41))
sd = 0.1
for i in range(41):
    for j in range(41):
        inter_mat[j,i] = np.exp(-(((x_train[j]-x_train[i])**2)/(2*(sd**2))))
        
weights = np.dot(np.linalg.pinv(inter_mat), y_train)

result_mat = np.zeros((201,41))
for j in range(201):
    for i in range(41):
        result_mat[j,i] = np.exp(-(((x_test[j]-x_train[i])**2)/(2*(sd**2))))
y_pred = np.dot(result_mat, weights)

plt.plot(x_train, y_train)
plt.plot(x_test, y_test, color = 'red', label = 'actual value')
plt.plot(x_test, y_pred, color = 'blue', label = 'prediction')
plt.xlabel('x value')
plt.ylabel('function value')
plt.title('Function Approximation')
plt.legend(loc = 'upper left')
plt.savefig('Q1a.png')
plt.show()
