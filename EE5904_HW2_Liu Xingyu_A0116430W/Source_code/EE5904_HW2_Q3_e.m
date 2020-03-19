clc;
clear all;
%% sequential mode training
for i = [1:10,20,50];
[acc_train5, acc_val5] = sequ(i,20);
fprintf('Accuracy of %d hidden neutrons for training data is %f.\n', i, acc_train5);
fprintf('Accuracy of %d hidden neutrons for validation data is %f.\n', i, acc_val5);
end
%%
function [acc_train5, acc_val5] = sequ(n,epochs)
V_train = [];
I_train = dir('group_3/train');
for i = 3:503
    im_train = double(imread(append('group_3/train/',I_train(i).name)));
    im_train = im_train(:);
    V_train = cat(2, V_train, im_train);
end

L_train = [];
for i = 3:503
    tmp = strsplit(I_train(i).name, {'_', '.'});
    L= str2num(tmp{2});
    L_train = cat(2, L_train, L);
end
%%
V_val = [];
I_val = dir('group_3/val');
for i = 3:169
    im_val = double(imread(append('group_3/val/',I_val(i).name)));
    im_val = im_val(:);
    V_val = cat(2, V_val, im_val);
end

L_val = [];
for i = 3:169
    tmp = strsplit(I_val(i).name, {'_', '.'});
    L= str2num(tmp{2});
    L_val = cat(2, L_val, L);
end
%%

% 1. Change the input to cell array form for sequential training
images_c = num2cell(V_train, 1);
labels_c = num2cell(L_train, 1);
    
% 2. Construct and configure the MLP
net = patternnet(n);
    
net.divideFcn = 'dividetrain';
net.performParam.regularization = 0.25;
net.trainFcn = 'traingdx';
net.trainParam.epochs = epochs;
    
accu_train = zeros(epochs,1);
accu_val = zeros(epochs,1);

train_num = 501;

for i = 1: epochs
%display(['Epoch: ', num2str(i)])
idx = randperm(train_num);
        
net = adapt(net, images_c(:,idx), labels_c(:,idx));
        
pred_train=round(net(V_train(:,1:train_num)));
accu_train(i) = 1 - mean(abs(pred_train-L_train(1:train_num)));
pred_val=round(net(V_train(:,train_num+1:end)));
accu_val(i) = 1 - mean(abs(pred_val-L_train(train_num+1:end)));
end
len_train = length(L_train);
len_val = length(L_val);
y_train = net(V_train);
y_train = (y_train >= 0.5);
acc_train5 = sum(y_train == L_train)/len_train;
y_val = net(V_val);
y_val = (y_val >= 0.5);
acc_val5 = sum(y_val == L_val)/len_val;

end
