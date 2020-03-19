#%% Q3c
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat("MNIST_database.mat")

# The last two digit of matric number (A0116430W) are 3 and 0
train = np.array((data['train_classlabel'] == 3) | (data['train_classlabel'] == 0)).squeeze()
test = np.array((data['test_classlabel'] == 3) | (data['test_classlabel'] == 0)).squeeze()

train = np.invert(train)
test = np.invert(test)

x_train = data['train_data'][:,train].transpose()
x_test = data['test_data'][:,test].transpose()
y_train = data['train_classlabel'][:,train].squeeze()
y_test = data['test_classlabel'][:,test].squeeze()

#%% Parameters setting
M = 100
ite = 1000
w = np.random.rand(M,784)
lr_init = 0.1
sigma_init = np.sqrt(10**2+10**2)/2
tc = ite/np.log(sigma_init)

#%% Training
for i in range(ite):
    for j in range(len(x_train)):
        sigma = sigma_init*np.exp(-i/tc)
        lr = lr_init*np.exp(-i/ite)
        euc_dis = np.sqrt(np.sum((w-x_train[j,:])**2,axis=1))
        i_x = np.argmin(euc_dis)
        index_win = i_x%10
        col_win = (i_x//10)+1
        for k in range(len(w)):
            index = k%10
            col = (k//10)+1
            d = np.sqrt((index_win-index)**2+(col_win-col)**2)
            h = np.exp(-d**2/(2*sigma**2))
            w[k,:] = w[k,:] + lr*h*(x_train[j,:]-w[k,:])
    print (i+1, "iteration completed")            

#%% Plotting c-1
fig, ax = plt.subplots(10, 10, figsize=(25,25))
for i in range(10):
    for j in range(10):
        index = j+i*10
        img = np.reshape(w[index,:],(28,28)).transpose()
        ax[i,j].imshow(img)
fig.savefig('Q3c1')
#%% Predict the testing data c-2
w_label = []
for i in range(100):
    dis2 = np.sqrt(np.sum((w[i,:]-x_train)**2,axis=1))
    ind_dist2 = np.argmin(dis2)
    w_label.append(y_train[ind_dist2])
    
w_label = np.array(w_label)

y_pred = []
for j in range(len(x_test)):
    dis3 = np.sqrt(np.sum((w-x_test[j,:])**2,axis=1))
    ind_dis3 = np.argmin(dis3)
    y_pred.append(w_label[ind_dis3])

y_pred = np.array(y_pred)

acc_test = (np.sum(y_pred == y_test))/len(y_test)

print('The accuracy of SOM on testing data is:',acc_test)
