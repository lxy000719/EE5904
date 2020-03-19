clc;
clear all;
%% transform the training data
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
%% transform the validation data
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
%% train the net
net = perceptron;
[net, tr] = train(net, V_train, L_train);
len_train = length(L_train);
len_val = length(L_val);
y_train = net(V_train);
y_train = (y_train >= 0.5);
acc_train = sum(y_train == L_train)/len_train;
y_val = net(V_val);
y_val = (y_val >= 0.5);
acc_val = sum(y_val == L_val)/len_val;
%% print the accuracy
fprintf('Accuracy of single perceptron for training data is %f.\n', acc_train);
fprintf('Accuracy of single perceptron for validation data is %f.\n', acc_val);
