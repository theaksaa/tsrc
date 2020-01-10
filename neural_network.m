clear

traffic_signs_images_train = fullfile('D:','Traffic signs recognition and classification', 'Train');
%traffic_signs_images_validation = fullfile('D:','Traffic signs recognition and classification', 'Train_resized');

imds = imageDatastore(traffic_signs_images_train, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%imdsValidation = imageDatastore(traffic_signs_images_validation, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%labelCount = countEachLabel(imdsValidation)

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomize');

layers = [
    imageInputLayer([32 32 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(43)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 50, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(imdsTrain, layers, options, 'NegativeOverlapRange', [0 0.3]);
save net;

% load net;
% 
% test = imread('test.png');
% test = imresize(test, [32 32]);
% YPred = classify(net, test)

%YValidation = imdsValidation.Labels;

%accuracy = sum(YPred == YValidation)/numel(YValidation)
