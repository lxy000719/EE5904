clear all;
clc;
%% without regularization
i = 20;
[acc_train3, acc_val3] = withoutpca(i);
fprintf('Accuracy of %d hidden neutrons for training data without regularization is %f.\n', i, acc_train3);
fprintf('Accuracy of %d hidden neutrons for validation data without regularization is %f.\n', i, acc_val3);

%%
function [acc_train3, acc_val3] = withoutpca(n)
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
net = patternnet(n);

[net, tr] = train(net, V_train, L_train);
len_train = length(L_train);
len_val = length(L_val);
y_train = net(V_train);
y_train = (y_train >= 0.5);
acc_train3 = sum(y_train == L_train)/len_train;
y_val = net(V_val);
y_val = (y_val >= 0.5);
acc_val3 = sum(y_val == L_val)/len_val;
end
