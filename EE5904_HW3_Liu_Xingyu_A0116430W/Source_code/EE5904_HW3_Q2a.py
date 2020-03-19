#%% Q2a
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

# %%No regularization
inter_mat = np.zeros((len(x_train),len(x_train)))
sd = 100

for i in range(len(x_train)):
    for j in range(len(x_train)):
        inter_mat[i,j] = np.exp(-(((np.linalg.norm(x_train[i]-x_train[j]))**2)/(2*(sd**2))))
        
weights = np.dot(np.linalg.pinv(inter_mat), y_train)

result_mat_test = np.zeros((len(x_test),len(x_train)))
for i in range(len(x_test)):
    for j in range(len(x_train)):
        result_mat_test[i,j] = np.exp(-(((np.linalg.norm(x_test[i]-x_train[j]))**2)/(2*(sd**2))))

result_mat_train = np.zeros((len(x_train),len(x_train)))
for i in range(len(x_train)):
    for j in range(len(x_train)):
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
plt.title('Accuracy versus thereshold with no regularization')
plt.legend()
plt.savefig('Q2a with no reg.png')
plt.show()

#%% With regularization
reg_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]

for reg in reg_list:
    inter_mat = np.zeros((len(x_train),len(x_train)))
    sd = 100
    
    I = np.identity(len(x_train))
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            inter_mat[i,j] = np.exp(-(((np.linalg.norm(x_train[i]-x_train[j]))**2)/(2*(sd**2))))
        
    weights = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(inter_mat), inter_mat)+reg*I), np.transpose(inter_mat)), y_train)

    result_mat_test = np.zeros((len(x_test),len(x_train)))
    for i in range(len(x_test)):
        for j in range(len(x_train)):
            result_mat_test[i,j] = np.exp(-(((np.linalg.norm(x_test[i]-x_train[j]))**2)/(2*(sd**2))))

    result_mat_train = np.zeros((len(x_train),len(x_train)))
    for i in range(len(x_train)):
        for j in range(len(x_train)):
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
    plt.title('Accuracy versus thereshold with regularization {}'.format(reg))
    plt.legend()
    plt.savefig('Q2a with reg {}.png'.format(reg))
    plt.show()
    
