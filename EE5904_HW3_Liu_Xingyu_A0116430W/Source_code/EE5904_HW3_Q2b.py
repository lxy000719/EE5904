#%% Q2b
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

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

# %% Fix dismax
M = 100
index = np.random.choice(len(x_train),M)
centers = np.ones((M,784))
for i in range(M):
    centers[i,:] = x_train[index[i],:]
    
dis = []
for i in range(M):
    for j in range(M):
        dis.append(np.linalg.norm(centers[i]-centers[j]))
dismax = np.max(dis)
sd = dismax / np.sqrt(2*M)

inter_mat = np.zeros((len(x_train),M))

for i in range(len(x_train)):
    for j in range(M):
        inter_mat[i,j] = np.exp(-(((np.linalg.norm(x_train[i]-x_train[j]))**2)/(2*(sd**2))))
        
weights = np.dot(np.linalg.pinv(inter_mat), y_train)

result_mat_test = np.zeros((len(x_test),M))

for i in range(len(x_test)):
    for j in range(M):
        result_mat_test[i,j] = np.exp(-(((np.linalg.norm(x_test[i]-x_train[j]))**2)/(2*(sd**2))))

result_mat_train = np.zeros((len(x_train),M))

for i in range(len(x_train)):
    for j in range(M):
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
plt.title('Accuracy versus thereshold')
plt.legend()
plt.savefig('Q2b.png')
plt.show()
print('The dismax selected is', dismax)

# %% With different dismax from 0.1 to 10000
d = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]

for dismax in d:
    M = 100
    index = np.random.choice(len(x_train),M)
    centers = np.ones((M,784))
    
    for i in range(M):
        centers[i,:] = x_train[index[i],:]
    
    sd = dismax / np.sqrt(2*M)

    inter_mat = np.zeros((len(x_train),M))

    for i in range(len(x_train)):
        for j in range(M):
            inter_mat[i,j] = np.exp(-(((np.linalg.norm(x_train[i]-x_train[j]))**2)/(2*(sd**2))))
        
    weights = np.dot(np.linalg.pinv(inter_mat), y_train)

    result_mat_test = np.zeros((len(x_test),M))
    
    for i in range(len(x_test)):
        for j in range(M):
            result_mat_test[i,j] = np.exp(-(((np.linalg.norm(x_test[i]-x_train[j]))**2)/(2*(sd**2))))

    result_mat_train = np.zeros((len(x_train),M))
    
    for i in range(len(x_train)):
        for j in range(M):
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
    plt.title('Accuracy versus thereshold with different dismax {}'.format(dismax))
    plt.legend()
    plt.savefig('Q2b with dismax {}.png'.format(dismax))
    plt.show()
