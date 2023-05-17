%%  hog deskriptorje
clear all
path_background_train = 'C:\Users\cr008\OneDrive\Desktop\backgrounds_real_exp\backgrounds_real\train';
path_background_test = 'C:\Users\cr008\OneDrive\Desktop\backgrounds_real_exp\backgrounds_real\test';
path_tomato_train = 'C:\Users\cr008\OneDrive\Desktop\tomatoes_real_exp\tomatoes_real\train';
path_tomato_test = 'C:\Users\cr008\OneDrive\Desktop\tomatoes_real_exp\tomatoes_real\test';
%[D_train_hog_background, D_test_hog_background, D_train_param_hog_background, imPath_test_background]=HOG(path_background_train, path_background_test,'background');
load('D_test_hog_background.mat');
load('D_train_hog_background.mat');
load('D_train_param_hog_background.mat');
load('D_test_hog_tomato.mat');
load('D_train_hog_tomato.mat');
load('D_train_param_hog_tomato.mat');
load('imPath_test_background.mat');
load('imPath_test_tomato.mat');
load('gnd_truth_hog.mat');
load('predictions_hog.mat');

% zdruzene obe polovice ucne mnozice 
D_train_background_hog_celotna = [D_train_hog_background;D_train_param_hog_background];
D_train_tomato_hog_celotna = [D_train_hog_tomato;D_train_param_hog_tomato];

D_test_hog = [D_test_hog_tomato'; D_test_hog_background']';
D_train_hog = [D_train_tomato_hog_celotna', D_train_background_hog_celotna']';

[predictions_hog,gnd_truth_hog]= razvrscanje_HOG(D_train_hog, D_test_hog);
% FPR 0.1845, TPR 0.9005 z number of orientations 12
[TPR_hog,FPR_hog] = rate(gnd_truth_hog,predictions_hog);





