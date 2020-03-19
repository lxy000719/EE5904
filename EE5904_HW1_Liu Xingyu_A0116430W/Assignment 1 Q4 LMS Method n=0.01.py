"""LMS method"""
import numpy as np
import matplotlib.pyplot as plt

"""Input data including bias"""
X = np.array([[1,0],[1,0.8],[1,1.6],[1,3],[1,4],[1,5]])
"""Output data"""
d = np.array([0.5,1,4,5,6,9])
"""Learning rate"""
"""You can change the learning rate here to run different results"""
n = 0.01
"""Initial weight vector"""
w = np.array([1.0,1.0])
rows = np.shape(X)[0]
"""Initial error set to 0"""
e = 0
"""Set an array to store each changing weight along 100epochs"""
weight_array = np.empty([2,100])

"""100epochs using 0.01 learning rate"""
j=0
for j in range(100):
    i=0
    for i in range(rows):
        e = d[i]-np.dot(X[i,:],w)
        w = w + n*e*X[i,:]
        i = i+1
    weight_array[0,j]=w[0]
    weight_array[1,j]=w[1]
    j=j+1

print('Final weight after 100epochs is:')
print(w)

"""Plot the weight change along 100epochs and LMS fitting line"""
step = np.arange(100)
x = np.arange(6)
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].plot(step,weight_array[1,:],label='w')
axes[0].plot(step,weight_array[0,:],label='b')
axes[0].set_title('LMS weight change along 100 epochs')
axes[0].set_xlabel('epochs')
axes[0].set_ylabel('weight')
axes[0].legend(loc='upper left')
axes[1].plot(x, w[0]+w[1]*x,label = 'fitting line')
axes[1].scatter(X[:,1],d,color='green',label = 'input data point')
axes[1].set_title('LMS fitting line')
axes[1].set_xlabel('input x')
axes[1].set_ylabel('desired output d')
axes[1].legend(loc='upper left')
plt.tight_layout()

