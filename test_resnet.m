clear all;  close all;
addpath(genpath('dependencies/'));

setup; % set up the date dependencies
opts.imdb       = [];
opts.networkType = 'resnet' ;
opts.expDir = 'trained_models\wow_04';

%other parameter settings
opts.batchNormalization = true ;
opts.nClasses = 2;
opts.batchSize = 1; 
opts.numAugments = 1 ;
opts.numEpochs = 30; % this is the number of voting ensembles
opts.bn = true; % batch normalization or not 
opts.whitenData = true;
opts.contrastNormalization = true;
opts.meanType = 'image'; % 'pixel' | 'image'
opts.gpus = [];  % the index of gpu. 
opts.checkpointFn = [];

% load the trained model
fileName = 'trained_models\wow_04\wow_model.mat'; % load a well trained model for validation 
load(fileName, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;

file_add = 'test_images\'; % set the address of testing images
Num = 10; % set the number of testing images
accomp_cover = 'batch_images\covers\'; 
accomp_stego = 'batch_images\wow_04\'; % the accompany stego image should be same to the trained model
imdb = cnn_steganalysis_setup_data(file_add, Num, accomp_cover, accomp_stego); % set the data for processing
% -------------------------------------------------------------------------
%                                                                     Test
% -------------------------------------------------------------------------
testfn = @cnn_train_dag_check; 

[net, info] = testfn(net, imdb, getBatchFn(opts, net.meta), ... 
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  'gpus', opts.gpus, ...
  'batchSize',opts.batchSize,...
  'numEpochs',opts.numEpochs,...
  'val', find(imdb.images.set == 2), ...
  'derOutputs', {'loss', 1}, ...
  'checkpointFn', opts.checkpointFn) ;

