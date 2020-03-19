clc
clear all;
%% sampling points in the domain of [-1,1]
x=-1:0.05:1;
%% generating training data, and the desired outputs
y=1.2*sin(pi*x) - cos(2.4*pi*x);
%% specify the structure and learning algorithm for MLP
neuron_list = [1:10 20 50];

for n = neuron_list
net = feedforwardnet(n, 'traingdx');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
net = configure(net, x, y); 
    for epoch = 1: 1000
    idx = randperm(length(x));
    net = adapt(net, x(:,idx), y(:,idx));
    end
xtest=-1:0.01:1;
ytest=1.2*sin(pi*xtest) - cos(2.4*pi*xtest);
xtest33=-3:0.01:3;
ytest33=1.2*sin(pi*xtest33) - cos(2.4*pi*xtest33);
net_output=sim(net,xtest);
net_output33=sim(net,xtest33);
subplot(2,1,1)
plot(xtest,ytest,xtest,net_output,'o');
xlabel('x')
ylabel('y')
str1 = sprintf('Sequential mode, trainlm, ytest, %d neurons', n);
title(str1)
xlim([-3 3])
subplot(2,1,2)
plot(xtest33,ytest33,xtest33,net_output33,'o');
xlabel('x')
ylabel('y')
str2 = sprintf('Sequential mode, trainlm, prediction on interval -3 to 3, %d neurons', n);
title(str2)
saveas(gcf, ['Sequential mode, number of neurons ', num2str(n), ' trainlm'], 'png');
end
