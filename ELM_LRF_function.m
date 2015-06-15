function [ result ] = ELM_LRF_function( param_name, C )
% If use for NORB dataset, please download the NORB data from:
% http://web.stanford.edu/~asaxe/random_weights.html. Unzip it and put it
% in the same foler with the codes.

% load parameters
load([param_name '.mat']);

startup;
% load testing data
fprintf('Loading testing data...\n');
load D:\Dataset\random_weights_paper1\norbdata\norb_testdata;
X_t=X;
Y_t=Y;
clear X Y;
NumberofTestingData=length(Y_t);

% no need to preprocess the labels of testing data
% testing data: nonlinear transform through the sinle-layer convolutional
% ELM

% load training data
fprintf('Loading training data...\n');
load D:\Dataset\random_weights_paper1\norbdata\norb_traindata.mat;
NumberofTrainingData=length(Y);

% preprocessing of the labels of training data
fprintf('Preprocess the labels of training data...\n');
number_class=length(unique(Y));

temp_Y=-ones(number_class, NumberofTrainingData);
for i=1:NumberofTrainingData;
    temp_Y(Y(i), i)=1;
end

image_size=sqrt(size(X, 1)/input_ch);
num_examples=size(X, 2);

% Begins to generate random filters
fprintf('Begins to generate random filters...\n');
tic;

layer_param=param.network_params{1};
layer_param.input_ch=input_ch;
layer_param.image_size=image_size;

% randomly generate the weight matrix W
[W, rf_index, pool_index, h_dim, tied_units]=gen_weights_my2(layer_param);

% update parameters for the next layer
image_size=h_dim;
input_ch=size(rf_index, 1)/(image_size^2);

% forwardprop X through current layer to generate input for the next
% layer
W_temp=expand_rf(layer_param, h_dim, tied_units, W);
W_temp=full_size(W_temp, rf_index);
[dummy, X]=two_layer_forwardprop(X, W_temp, pool_index, layer_param.l1_act, layer_param.l2_act); 
time_network=toc;

% the training of the last layer (ELM-random) begins:
fprintf('The training of the last layer (ELM-random) begins:\n');

% calculating the output weight \beta and predict_Y
if size(X, 1)<=size(X, 2);
    tic;
    beta=(eye(size(X, 1))/C+X*X')\(X*temp_Y');
    predict_Y=(X'*beta)';
    train_time=toc;
else
    tic;
    beta=X*((eye(size(X, 2))/C+X'*X)\(temp_Y'));
    predict_Y=(X'*beta)';
    train_time=toc;
end

% calculate training classification accuracy
MissClassificationRate_Training=0;
for k=1:NumberofTrainingData;
    [dummy, label_expected]=max(predict_Y(:, k));
    if label_expected~=Y(k);
        MissClassificationRate_Training=MissClassificationRate_Training+1;
    end
end
train_accuracy=1-MissClassificationRate_Training/NumberofTrainingData

fprintf('The testing begins:\n');
tic;
for a=1:param.num_layers;
    [dummy, X_t]=two_layer_forwardprop(X_t, W_temp, pool_index, layer_param.l1_act, layer_param.l2_act);
end
time_transform_test_data=toc;

% calculating the predicted Y_t
tic;
predict_Y_t=(X_t'*beta)';
test_time=toc;

MissClassificationRate_Testing=0;
for k=1:NumberofTestingData;
    [dummy, label_expected]=max(predict_Y_t(:, k));
    if label_expected~=Y_t(k);
        MissClassificationRate_Testing=MissClassificationRate_Testing+1;
    end
end
test_accuracy=1-MissClassificationRate_Testing/NumberofTestingData

result=struct;
result.train_time=train_time;
result.test_time=test_time;
result.train_accuracy=train_accuracy;
result.test_accuracy=test_accuracy;
%result.OutputWeight=beta;
result.time_network=time_network;
result.time_transform_test_data=time_transform_test_data;

end