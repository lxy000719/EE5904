#%% Q3a
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-np.pi,np.pi,200)
x_train = np.array([t*np.sin(np.pi*np.sin(t)/t), 1-np.abs(t)*np.cos(np.pi*np.sin(t)/t)])
x_train = x_train.transpose()

M = 25
ite = 500
w = np.random.rand(M,2)
lr_init = 0.1
sigma_init = np.sqrt(25**2+1**2)/2
tc = ite/np.log(sigma_init)

for i in range(ite):
    for j in range(len(x_train)):
        sigma = sigma_init*np.exp(-i/tc)
        lr = lr_init*np.exp(-i/ite)
        dis = np.sqrt(np.sum((w-x_train[j,:])**2,axis=1))
        ind_win = np.argmin(dis)
        for k in range(len(w)):
            d = np.abs(k-ind_win)
            h = np.exp(-d**2/(2*sigma**2))
            w[k,:] = w[k,:] + lr*h*(x_train[j,:]-w[k,:])
    print(i+1,'iteration completed')
     
#%% Plotting       
plt.scatter(w[:,0],w[:,1])
plt.plot(x_train[:,0],x_train[:,1])

plt.plot(w[:,0],w[:,1],color='red')
plt.plot(w[[0,24],0],w[[0,24],1],color='red')
plt.savefig('Q3a.png')

plt.show()