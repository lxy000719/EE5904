%% Q1. Rosenbrock's Valley Problem
%% a). Learning rate = 0.001
clc;
clear;

lr = 0.001;
w = zeros(2,2);
w(1,1) = rand(1) * 0.5;
w(1,2) = rand(1) * 0.5;
devx = zeros(2,1);
devy = zeros(2,1);
f = zeros(2,1);
f(1) = (1-w(1,1))^2+100*((w(1,2)-(w(1,1)^2))^2);

iter = 1;
iter_list = zeros(2,1);
iter_list(1) = 1;
while f(iter) > 0.0001
    devx(iter) = 400*(w(iter,1)^3)-400*w(iter,1)*w(iter,2)+2*w(iter,1)-2;
    devy(iter) = 200*w(iter,2)-200*(w(iter,1)^2);
    w(iter+1,1) = w(iter,1)-lr*devx(iter);
    w(iter+1,2) = w(iter,2)-lr*devy(iter);
    iter = iter + 1;
    iter_list(iter) = iter;
    f(iter) = (1-w(iter,1))^2+100*((w(iter,2)-(w(iter,1)^2))^2);
end

fprintf('The training ends at %f iteration and when function value is less than 0.0001\n',iter);
fprintf('The final x is %f and y is %f and function value is %f\n', [w(iter,1),w(iter,2),f(iter,1)]);

subplot(2,1,1);
plot(w(:,1),w(:,2));
xlabel('x');
ylabel('y');
title('x and y trajectory when learning rate is 0.001')

subplot(2,1,2);
plot(iter_list,f);
xlabel('Iteration');
ylim([0 1.5])
ylabel('f value');
title('Function value versus iteration')

%% a). Learning rate = 0.2
clc;
clear;

lr = 0.2;
w = zeros(2,2);
w(1,1) = rand(1) * 0.5;
w(1,2) = rand(1) * 0.5;
devx = zeros(2,1);
devy = zeros(2,1);
f = zeros(2,1);
f(1) = (1-w(1,1))^2+100*((w(1,2)-(w(1,1)^2))^2);

iter = 1;
iter_list = zeros(2,1);
iter_list(1) = 1;
while f(iter) > 0.0001
    devx(iter) = 400*(w(iter,1)^3)-400*w(iter,1)*w(iter,2)+2*w(iter,1)-2;
    devy(iter) = 200*w(iter,2)-200*(w(iter,1)^2);
    w(iter+1,1) = w(iter,1)-lr*devx(iter);
    w(iter+1,2) = w(iter,2)-lr*devy(iter);
    iter = iter + 1;
    iter_list(iter) = iter;
    f(iter) = (1-w(iter,1))^2+100*((w(iter,2)-(w(iter,1)^2))^2);
end

fprintf('The training ends at %f iteration and when function value will diverge to infinity when learning rate is 0.2\n',iter);

subplot(2,1,1);
plot(w(:,1),w(:,2));
xlabel('x');
ylabel('y');
title('x and y trajectory when learning rate is 0.2')

subplot(2,1,2);
plot(iter_list,f);
xlabel('Iteration');
ylabel('f value');
title('Function value versus iteration')

%% b). Newton's method
clc;
clear;

w = zeros(2,2);
w(1,1) = rand(1) * 0.5;
w(1,2) = rand(1) * 0.5;
devx = zeros(2,1);
devy = zeros(2,1);
H = zeros(2,2);
f = zeros(2,1);
f(1) = (1-w(1,1))^2+100*((w(1,2)-(w(1,1)^2))^2);

iter = 1;
iter_list = zeros(2,1);
iter_list(1) = 1;
while f(iter) > 0.0001
    devx(iter) = 400*(w(iter,1)^3)-400*w(iter,1)*w(iter,2)+2*w(iter,1)-2;
    devy(iter) = 200*w(iter,2)-200*(w(iter,1)^2);
    H = [1200*(w(iter,1)^2)-400*w(iter,2)+2 -400*w(iter,1); -400*w(iter,1) 200];
    temp = [w(iter,1);w(iter,2)] - inv(H)*[devx(iter);devy(iter)];
    w(iter+1,1) = temp(1);
    w(iter+1,2) = temp(2);
    iter = iter + 1;
    iter_list(iter) = iter;
    f(iter) = (1-w(iter,1))^2+100*((w(iter,2)-(w(iter,1)^2))^2);
end

fprintf('The training ends at %f iteration and when function value is less than 0.0001\n',iter);
fprintf('The final x is %f and y is %f and function value is %f\n', [w(iter,1),w(iter,2),f(iter,1)]);

subplot(2,1,1);
plot(w(:,1),w(:,2));
xlabel('x');
ylabel('y');
title('x and y trajectory using Newtons method')

subplot(2,1,2);
plot(iter_list,f);
xlabel('Iteration');
ylabel('f value');
title('Function value versus iteration')