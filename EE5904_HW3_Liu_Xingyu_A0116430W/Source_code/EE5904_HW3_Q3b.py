#%% Q3b
import numpy as np
import matplotlib.pyplot as plt

x_train = np.random.rand(2,500)
x_train = x_train.transpose()

M = 25
ite = 500
w = np.random.rand(M,2)
lr_init = 0.1
sigma_init = np.sqrt(5**2+5**2)/2
tc = ite/np.log(sigma_init)

for i in range(ite):
    for j in range(len(x_train)):
        sigma = sigma_init*np.exp(-i/tc)
        lr = lr_init*np.exp(-i/ite)
        dis = np.sqrt(np.sum((w-x_train[j,:])**2,axis=1))
        ind_win = np.argmin(dis)
        index_win = ind_win%5
        col_win = (ind_win//5)+1
        for k in range(len(w)):
            index = k%5
            col = (k//5)+1
            d = np.sqrt((index_win-index)**2+(col_win-col)**2)
            h = np.exp(-d**2/(2*sigma**2))
            w[k,:] = w[k,:] + lr*h*(x_train[j,:]-w[k,:])
    print(i+1,'iteration completed')

#%% Plotting            
plt.scatter(x_train[:,0],x_train[:,1],label = 'x_train')
plt.scatter(w[:,0],w[:,1],color = 'red',label = 'SOM')

plt.plot(w[0:5,0],w[0:5,1],color = 'red')
plt.plot(w[5:10,0],w[5:10,1],color = 'red')
plt.plot(w[10:15,0],w[10:15,1],color = 'red')
plt.plot(w[15:20,0],w[15:20,1],color = 'red')
plt.plot(w[20:25,0],w[20:25,1],color = 'red')
plt.plot(w[[0,5,10,15,20],0],w[[0,5,10,15,20],1],color = 'red')
plt.plot(w[[1,6,11,16,21],0],w[[1,6,11,16,21],1],color = 'red')
plt.plot(w[[2,7,12,17,22],0],w[[2,7,12,17,22],1],color = 'red')
plt.plot(w[[3,8,13,18,23],0],w[[3,8,13,18,23],1],color = 'red')
plt.plot(w[[4,9,14,19,24],0],w[[4,9,14,19,24],1],color = 'red')
plt.title('x_train and SOM map')

plt.legend()
plt.savefig('Q3b.png')
plt.show()