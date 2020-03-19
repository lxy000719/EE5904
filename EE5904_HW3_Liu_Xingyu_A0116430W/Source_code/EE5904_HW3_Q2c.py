#%% Q2c
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans

data = loadmat("MNIST_database.mat")

# The last two digit of matric number (A0116430W) are 3 and 0
train = np.array((data['train_classlabel'] == 3) | (data['train_classlabel'] == 0)).squeeze()
test = np.array((data['test_classlabel'] == 3) | (data['test_classlabel'] == 0)).squeeze()
x_train = data['train_data'][:,train].transpose()
x_test = data['test_data'][:,test].transpose()
y_train = data['train_classlabel'][:,train].squeeze()
y_test = data['test_classlabel'][:,test].squeeze()
y_train = (y_train == 3).astype(int)
y_test = (y_test == 3).astype(int)

#%%
kmean = KMeans(n_clusters=2)
kmean.fit(x_train)
centers = kmean.cluster_centers_
n_clusters = 2

dismax = 100
sd = dismax / np.sqrt(2*n_clusters)

inter_mat = np.zeros((len(x_train),n_clusters))

for i in range(len(x_train)):
    for j in range(n_clusters):
        inter_mat[i,j] = np.exp(-(((np.linalg.norm(x_train[i]-x_train[j]))**2)/(2*(sd**2))))
        
weights = np.dot(np.linalg.pinv(inter_mat), y_train)

result_mat_test = np.zeros((len(x_test),n_clusters))

for i in range(len(x_test)):
    for j in range(n_clusters):
        result_mat_test[i,j] = np.exp(-(((np.linalg.norm(x_test[i]-x_train[j]))**2)/(2*(sd**2))))

result_mat_train = np.zeros((len(x_train),n_clusters))

for i in range(len(x_train)):
    for j in range(n_clusters):
        result_mat_train[i,j] = np.exp(-(((np.linalg.norm(x_train[i]-x_train[j]))**2)/(2*(sd**2))))

y_pred_test = np.dot(result_mat_test, weights)
y_pred_train = np.dot(result_mat_train, weights)

TrAcc = np.zeros((1000,1)).squeeze()
TeAcc = np.zeros((1000,1)).squeeze()
thr = np.zeros((1000,1)).squeeze()
TrN = len(y_train)
TeN = len(y_test)
    
for i in range(1000):
    t = (np.max(y_pred_train)-np.min(y_pred_train)) * i/1000 + np.min(y_pred_train)
    thr[i] = t
    TrAcc[i] = np.sum((y_pred_train>=t)==y_train)/len(y_pred_train)
    TeAcc[i] = np.sum((y_pred_test>=t)==y_test)/len(y_pred_test)

plt.plot(thr,TrAcc,label='Training accuracy')
plt.plot(thr,TeAcc,label='Testing accuracy')
plt.xlabel('Thereshold')
plt.ylabel('Accuracy')
plt.title('Accuracy versus thereshold with K-means clustering')
plt.legend()
plt.savefig('Q2c1.png')
plt.show()

#%%
fig, ax = plt.subplots(2, 2)
ax[0,0].imshow(np.reshape(centers[0], (28,28)).transpose())
ax[0,1].imshow(np.reshape(centers[1], (28,28)).transpose())
ax[1,0].imshow(np.reshape(np.mean(x_train[y_train==0], axis=0), (28,28)).transpose())
ax[1,1].imshow(np.reshape(np.mean(x_train[y_train==1], axis=0), (28,28)).transpose())
ax[0,0].set_title('K-means cluster center 0')
ax[0,1].set_title('K-means cluster center 3')
ax[1,0].set_title('Training data mean 0')
ax[1,1].set_title('Training data mean 3')
plt.tight_layout()
plt.savefig('Q2c2.png')
plt.show()
