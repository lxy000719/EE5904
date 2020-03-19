clc;
clear all;
%%
[acc_train, acc_val] = withpca(0.5);
fprintf('Accuracy of single perceptron for training data with pca of resize 128x128 is %f.\n', acc_train);
fprintf('Accuracy of single perceptron for validation data with pca of resize 128x128 is %f.\n', acc_val);
[acc_train, acc_val] = withpca(0.25);
fprintf('Accuracy of single perceptron for training data with pca of resize 64x64 is %f.\n', acc_train);
fprintf('Accuracy of single perceptron for validation data with pca of resize 64x64 is %f.\n', acc_val);
[acc_train, acc_val] = withpca(0.125);
fprintf('Accuracy of single perceptron for training data with pca of resize 32x32 is %f.\n', acc_train);
fprintf('Accuracy of single perceptron for validation data with pca of resize 32x32 is %f.\n', acc_val);

[acc_train2, acc_val2] = withoutpca(0.5);
fprintf('Accuracy of single perceptron for training data without pca of resize 128x128 is %f.\n', acc_train2);
fprintf('Accuracy of single perceptron for validation data without pca of resize 128x128 is %f.\n', acc_val2);
[acc_train2, acc_val2] = withoutpca(0.25);
fprintf('Accuracy of single perceptron for training data without pca of resize 64x64 is %f.\n', acc_train2);
fprintf('Accuracy of single perceptron for validation data without pca of resize 64x64 is %f.\n', acc_val2);
[acc_train2, acc_val2] = withoutpca(0.125);
fprintf('Accuracy of single perceptron for training data without pca of resize 32x32 is %f.\n', acc_train2);
fprintf('Accuracy of single perceptron for validation data without pca of resize 32x32 is %f.\n', acc_val2);
%%
function [acc_train, acc_val] = withpca(resize)
V_train = [];
I_train = dir('group_3/train');
for i = 3:503
    im_train = pca(imresize(double(imread(append('group_3/train/',I_train(i).name))),resize));
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
    im_val = pca(imresize(double(imread(append('group_3/val/',I_val(i).name))),resize));
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
net = perceptron;
[net, tr] = train(net, V_train, L_train);
y_train = net(V_train);
perf_train = perform(net, L_train, y_train); 
acc_train = 1 - perf_train;
y_val = net(V_val);
perf_val = perform(net, L_val, y_val); 
acc_val = 1 - perf_val;
end


%%
function [acc_train2, acc_val2] = withoutpca(resize)
V_train = [];
I_train = dir('group_3/train');
for i = 3:503
    im_train = imresize(double(imread(append('group_3/train/',I_train(i).name))),resize);
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
    im_val = imresize(double(imread(append('group_3/val/',I_val(i).name))),resize);
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
net = perceptron;
[net, tr] = train(net, V_train, L_train);
len_train = length(L_train);
len_val = length(L_val);
y_train = net(V_train);
y_train = (y_train >= 0.5);
acc_train2 = sum(y_train == L_train)/len_train;
y_val = net(V_val);
y_val = (y_val >= 0.5);
acc_val2 = sum(y_val == L_val)/len_val;
end
