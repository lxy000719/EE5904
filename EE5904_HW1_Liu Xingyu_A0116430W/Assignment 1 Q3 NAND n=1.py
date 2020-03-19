import numpy as np
import matplotlib.pyplot as plt

"""NAND Implementation"""
X = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
y = np.array([1,1,1,0])

"""Initial weight"""
w = np.array([0.8,0.8,0.8])

"""Learning rate"""
"""You can change the learning rate here to run different results"""
n = 1

"""4 iteration in 1 epoch"""
rows = np.shape(X)[0]

"""To store every changing w"""
weight_array = np.copy(w)

"""Iteration of changing w"""
"""There are total 4x10=40 iterations for changing w"""
j = 0
v = 0
for j in range(10):
    i = 0
    for i in range(rows):
        v = np.dot(X[i,:],w)
        if v >= 0:
            v = 1
        else:
            v = 0
        e = y[i]-v
        w = w + n*e*X[i,:]
        weight_array = np.append(weight_array,w)
        i = i+1
        

total_iter = i*(j+1)
weight_array = np.reshape(weight_array, (total_iter+1,3))

b = w[1]/w[2]
c = w[0]/w[2]

"""Plot the NAND Implementation and weight changing trajectories versus iteration"""
a = np.arange(3)    
step = np.arange(total_iter+1)
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].plot(a,-(b*a)-c)
axes[0].scatter(0,0,color = 'green',label = 'class y =1')
axes[0].scatter(0,1,color = 'green',label = 'class y =1')
axes[0].scatter(1,0,color = 'green',label = 'class y =1')
axes[0].scatter(1,1,color = 'red',label = 'class y =0')
axes[0].legend(loc = 'best')
axes[0].set_xlabel('Value of x1')
axes[0].set_ylabel('Value of x2')
axes[0].set_title('NAND Implementation')
axes[1].plot(step, weight_array[:,0],label = 'weight change for bias')
axes[1].plot(step, weight_array[:,1],label = 'weight change for x1')
axes[1].plot(step, weight_array[:,2],label = 'weight change for x2')
axes[1].legend(loc = 'best')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Weight value')

