"""LLS Method"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

"""Input data including bias"""
X = np.array([[1,0],[1,0.8],[1,1.6],[1,3],[1,4],[1,5]])
"""Output data"""
d = np.array([0.5,1,4,5,6,9])

"""X transpose"""
X_trans = np.transpose(X)
"""Inverse of dot product of X transpose and X"""
X_trans_inv = inv(np.dot(X_trans,X))
"""LLS calculation of weight"""
temp = np.dot(X_trans_inv,X_trans)
w = np.dot(temp,d)

"""Plot using the calculated w"""
x = np.arange(6)
fig = plt.figure(figsize=(8,4))
graph = fig.add_axes([0,0,1,1])
graph.set_xlabel('Input')
graph.set_ylabel('Output')
graph.plot(x, w[1]*x+w[0],color='green', label='LLS fitting line')
graph.scatter(X[:,1],d,label='Input data point')
graph.legend(loc='upper left')

print('The final weight vector is:')
print(w)


