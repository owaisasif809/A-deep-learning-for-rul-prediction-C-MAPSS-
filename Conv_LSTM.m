%% Create a directory to store the Turbofan Engine Degradation Simulation data set.
clc, close all, clear all
dataFolder = fullfile('D:\MS EE\Predictive Maintenance\Datasets',"turbofan");
if ~exist(dataFolder,'dir')
    mkdir(dataFolder);
end
filename = "CMAPSSData.zip";
unzip(filename,dataFolder)

%% Prepare Training Data             %% (STEP # 1)
filenamePredictors = fullfile(dataFolder,"train_FD001.txt");
[XTrain,YTrain] = processTurboFanDataTrain(filenamePredictors);
%  for i = 1 :100
plot((XTrain{1}(3,:)))
%  pause(1)
%  end
title("Training Data, First Observation")
hold on
plot(YTrain{2}')
title("Labels")


%% Correltaion Analysis          %% (STEP # 2)
FD001_cc_d =zeros(24,1);
engines = 100;       % engines  , FD001=100, FD002=260, FD003=100, Fd004=249
for c = 1:engines    
    FD001_cc = [ ];
for b = 1:24     % sensor
  cc = corrcoef(XTrain{c}(b,:),YTrain{c}');
  new_cc = cc(2,1)';
  if (new_cc < 0)
      new_cc =-new_cc;
  end
  if isnan(new_cc)
     new_cc =0; 
  end
  FD001_cc = [FD001_cc; new_cc];
end
FD001_cc = FD001_cc * 100;             % convert into percnetage
FD001_cc_d = FD001_cc_d + FD001_cc;    % summing for average value
end
FD001_cc_d = FD001_cc_d/engines;        % average value of 24 sensor values

xvalues = {'op-1','op-2','op-3','s-1','s-2','s-3','s-4','s-5','s-6','s-7'...
     ,'s-8','s-9','s-10','s-11','s-12','s-13','s-14','s-15','s-16','s-17'...
     ,'s-18','s-19','s-20','s-21'};
yvalues = {'RUL'};
subplot(142)
ht = heatmap(yvalues, xvalues, FD001_cc_d);
ht.Title ='Heat map of FD002';
ht.XLabel = 'Remaining Useful Life';
ht.YLabel = 'Sensor Data';
ht.ColorLimits = [0 100];
ht.FontColor = 'Black';
ht.MissingDataLabel = '0'
ht.FontSize = 12;
%hold on
%pause (1);
%% Remove Un-correlared Data
% FD001
% FD001 = < 10,   FD002 = < 5, FD003= < 10, FD004 = < 5 
idxConstant = FD001_cc_d < 10;   
for i = 1:numel(XTrain)
    XTrain{i}(idxConstant,:) = [];
end

%% Prepare Data for Padding
% To minimize the amount of padding added to the mini-batches, sort the training 
% data by sequence length. Then, choose a mini-batch size which divides the training 
% data evenly and reduces the amount of padding in the mini-batches.
% Sort the training data by sequence length.
for i=1:numel(XTrain)
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
XTrain = XTrain(idx);
YTrain = YTrain(idx);
for  i =1:90                      % 234=FD002, 90=FD001,FD003, 224=FD004
    bb{i}=XTrain{i};
    cc{i}=YTrain{i};
end
for k= 1:10                     % 26=FD002,    10=FD001,FD003, 25=FD004
    j=k+90;                        
    valx{k}=XTrain{j};
    valy{k}=YTrain{j};  
end
XTrain=bb';
YTrain=cc';
valx=valx';
valy=valy';
figure
bar(sequenceLengths)
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")

%% Data Normalization

mu = mean([XTrain{:}],2);
sig = std([XTrain{:}],0,2);

for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - mu) ./ sig;
end

thr = 97; % for rul limit to 150
for i = 1:numel(YTrain)
    YTrain{i}(YTrain{i} > thr) = thr;
end
%% Data Preparation for Convolution Neural Network
XTrain_data =[];
YTrain_label =[];

for d = 1 : 100
data_s = size (XTrain{d},2);
dim = rem(data_s,5);

   for xsam=1:(data_s-dim)/5
     XSam{xsam} = XTrain{d}(:,(5*xsam)-4:5*xsam);
     YSam{xsam} = YTrain{d}((xsam*5));
   end
%xsam = xsam + (data_s-dim)/5;   
%XSam=XSam';
XTrain_data = [XTrain_data  XSam];
YTrain_label =[YTrain_label YSam];
clear XSam
clear YSam
end
XTrain_data = XTrain_data';
YTrain_label =YTrain_label';

%scaled_K = rescale(XSam);
%imshow(scaled_K)
%[S,F,T] = stft(XSam,2,'Window',kaiser(5,5),'OverlapLength',2);
%plot(T,abs(S(:,:,1)))
%plot(shtft)
%for ysam = 1:(data_s-dim)/5
%YSam{ysam} = YTrain{1}((ysam*5)+1);
%end

 %% Define LSTM Network Architecture
 % Define the network architecture. Create an LSTM network that consists of an LSTM layer
 % with 200 hidden units, followed by a fully connected layer of size 50 and a dropout layer
 % with dropout probability 0.5.
numResponses = size(YTrain{1},1);
numFeatures = size(XTrain{1},1);
numHiddenUnits = 200;
inputSize = [14 5 1];

layers = [ ...
    sequenceInputLayer([14 5 1],'Name','input')
    %imageInputLayer(inputSize,'Name','input')
    sequenceFoldingLayer('Name','fold')
    convolution2dLayer([1 3],20,'Name','conv')
    %batchNormalizationLayer('Name','bn')
    reluLayer('Name','relu')
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    lstmLayer(50,'OutputMode','sequence','Name','lstm')
    dropoutLayer(0.3,'Name','dpl')
    fullyConnectedLayer(40,'Name','ffl')
    dropoutLayer(0.5,'Name','dpl_2')
    fullyConnectedLayer(numResponses,'Name','ffl_2')
    regressionLayer('Name','regression')];

%lgraph = layerGraph(layers);
        lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');
plot(lgraph);
% To prevent the gradients from exploding, set the gradient threshold to 1. 
% To keep the sequences sorted by length, set 'Shuffle' to 'never'.
maxEpochs = 100;
miniBatchSize = 20;
% SGD = sgdm,      adam,      rmsprop
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',50, ...
    'LearnRateDropFactor',0.2, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);

% options = trainingOptions('adam', ...
%     'MaxEpochs',250, ...
%     'GradientThreshold',1, ...
%     'InitialLearnRate',0.005, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',125, ...
%     'LearnRateDropFactor',0.2, ...
%     'Verbose',0, ...
%     'Plots','training-progress');
%% Train the Network
net = trainNetwork(XTrain_data,YTrain_label,lgraph,options);
%net = trainNetwork(XTrain,YTrain,lgraph,options);

%%
%% Test The Network
filenamePredictors = fullfile(dataFolder,"test_FD001.txt");
filenameResponses = fullfile(dataFolder,"RUL_FD001.txt");
[XTest,YTest] = processTurboFanDataTest(filenamePredictors,filenameResponses);

for i = 1:numel(XTest)
    XTest{i}(idxConstant,:) = [];
end
for i = 1:numel(XTest)
    XTest{i} = (XTest{i} - mu) ./ sig;
    YTest{i}(YTest{i} > thr) = thr;
end
for i=1:numel(XTest)
    sequence = XTest{i};
    sequenceLengthss(i) = size(sequence,2);
end

[sequenceLengthss,idx] = sort(sequenceLengthss,'descend');
XTest = XTest(idx);
YTest = YTest(idx);

%% Data Preparation for Test Set
XTest_data =[];
YTest_label =[];

for dt = 1 : 100
data_st = size (XTest{dt},2);
dimt = rem(data_st,5);

   for xsamt=1:(data_st-dimt)/5
     XSamt{xsamt} = XTest{dt}(:,(5*xsamt)-4:5*xsamt);
     YSamt{xsamt} = YTest{dt}((xsamt*5));
   end
%xsam = xsam + (data_s-dim)/5;   
%XSam=XSam';
XTest_data = [XTest_data  XSamt];
YTest_label =[YTest_label YSamt];
clear XSamt
clear YSamt
end
XTest_data = XTest_data';
YTest_label =YTest_label';

YPred = predict(net,XTest_data,'MiniBatchSize',1);

    plot([YTest_label{:}],'--')
    hold on
    plot([YPred{:}],'.-')
    hold off
    
    title("Test Observation ")
    xlabel("Time Step")
    ylabel("RUL")
    
legend(["Test Data" "Predicted"],'Location','southeast')

%% RMS Error & Score Function for RUL Prediction
prev_label = 0;
for i = 1:100
    data_st = size (XTest{i},2);
    dimt = rem(data_st,5);
    label = (data_st-dimt)/5;
    prev_label = prev_label + label;
    YTestLast(i) = YTest_label{prev_label};
    YPredLast(i) = YPred{prev_label};
end
 %YTestLast = YTestLast';
 %YPredLast = YPredLast';
 
figure
rmse = sqrt(mean((YPredLast - YTestLast).^2))
histogram(YPredLast - YTestLast)
title("RMSE = " + rmse)
ylabel("Frequency")
xlabel("Error")

figure
%subplot(2,1,1)
plot(YTestLast)
hold on
plot(YPredLast,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("RUL")


