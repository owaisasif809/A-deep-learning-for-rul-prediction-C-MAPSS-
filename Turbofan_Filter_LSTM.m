  %% Create a directory to store the Turbofan Engine Degradation Simulation data set.
clc, close all, clear all
dataFolder = fullfile('D:\MS EE\Predictive Maintenance\Datasets',"turbofan");
if ~exist(dataFolder,'dir')
    mkdir(dataFolder);
end
filename = "CMAPSSData.zip";
%unzip(filename,dataFolder)


%% Prepare Training Data             %% (STEP # 1)
filenamePredictors = fullfile(dataFolder,"train_FD001.txt");    
[XTrain,YTrain] = processTurboFanDataTrain(filenamePredictors);  % split data into train & test
%  for i = 1 :100
%plot((XTrain{1}(3,:)))
%  pause(1)
%  end
%title("Training Data, First Observation")
%hold on
%plot(YTrain{2}')
%title("Labels")
%% Correltaion Analysis          %% (STEP # 2)
FD001_cc_d =zeros(24,1);
engines = 100;       % No of engines  , FD001=100, FD002=260, FD003=100, Fd004=249
for c = 1:engines    
    FD001_cc = [ ];
for b = 1:24     % total sensor in datset
  cc = corrcoef(XTrain{c}(b,:),YTrain{c}');   % correlation coefficient calculation
  new_cc = cc(2,1)';                           % single value from matrix
  if (new_cc < 0)                             % remove negative values
      new_cc =-new_cc;  
  end
  if isnan(new_cc)
     new_cc =0;                              % remove NaN number
  end
  FD001_cc = [FD001_cc; new_cc];     % concatenation into single matrix
end
FD001_cc = FD001_cc * 100;             % convert into percnetage
FD001_cc_d = FD001_cc_d + FD001_cc;    % summing for average value
end
FD001_cc_d = FD001_cc_d/engines;        % average value of 24 sensor values

xvalues = {'op-1','op-2','op-3','s-1','s-2','s-3','s-4','s-5','s-6','s-7'...
     ,'s-8','s-9','s-10','s-11','s-12','s-13','s-14','s-15','s-16','s-17'...
     ,'s-18','s-19','s-20','s-21'};
yvalues = {'RUL'};
% subplot(142)
% ht = heatmap(yvalues, xvalues, FD001_cc_d);
% ht.Title ='Heat map of FD002';
% ht.XLabel = 'Remaining Useful Life';
% ht.YLabel = 'Sensor Data';
% ht.ColorLimits = [0 100];
% ht.FontColor = 'Black';
% ht.MissingDataLabel = '0'
% ht.FontSize = 12;
%hold on
%pause (1);
%% Remove Un-correlared Data
% FD001   These threshold are also given in paper. sensor selection
% criteria
% FD001 = < 10,   FD002 = < 5, FD003= < 10, FD004 = < 5 
idxConstant = FD001_cc_d < 10;   
for i = 1:numel(XTrain)
    XTrain{i}(idxConstant,:) = [];
end

%% Filter and Normalize Training Predictors     %% (STEP # 3)
  % Moving Median Filter
SmoothData = XTrain;
for i = 1:numel(SmoothData)
    SmoothData{i} = SmoothData{i}'; % Transpose
end
for i = 1:numel(SmoothData)
    [SmoothData{i},window] = smoothdata(SmoothData{i},'movmedian');   % Moving Average Filter  'sgolay'= filter,'movmedian'
end
                                                                                       %    'rlowess'
for i = 1:numel(SmoothData)
    SmoothData{i} = SmoothData{i}'; % Transpose
end
XTrain = SmoothData;

%% min max normalization

%minVal = min([XTrain{:}],[],2); 
%maxVal = max([XTrain{:}],[],2);
%for i = 1:numel(XTrain)
%    XTrain{i} = (XTrain{i} - minVal) ./ ( maxVal - minVal );
%end

%% Data Normalization

mu = mean([XTrain{:}],2);
sig = std([XTrain{:}],0,2);

for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - mu) ./ sig;  % it is given in paper
end
%subplot(224)
%%
%plot(YTrain{2},XTrain{2})
%axis([0 400 -2 2])
%% Piece Wise Linear Function for RUL          %% (STEP # 4)
RUL = [];
window = 24; % it is given in experimental section.
e=0;
initial_RUL =0;
engine_RUL = [];
RUL_trend = [];
RUL_avg = [];
RUL_sensor=0;
FD001_RUL=0;
FD001_RUL_all = [];
for g=1:100  % engines FD001 = 100, FD002 = 260, FD003 = 100, FD004 = 249, % This alorithm is given in paper    

for f = 1:14  % sensor  FD001 = 14, FD002 = 24 , FD003 = 16 , FD004 = 24
    w1 = XTrain{g}(f, 1:window);
    w1= abs(w1);
for i= 1:floor(numel(YTrain{g})/window)-1 
    w2 = XTrain{g}(f , ((window)*i)+1:window*(i+1));
    w2= abs(w2);
    mean_1 = (mean(w1));
    mean_2 = (mean(w2));
    diff = abs(mean_1 - mean_2);
    RUL(i)= diff;
    if (diff>=0.2 && e==0)
       initial_RUL =numel(YTrain{g}) - window*i;  % RUL Calculation
       % engine_RUL = [engine_RUL ; initial_RUL];
       e=1;
    end    
end
e=0;
RUL_sensor = RUL_sensor + initial_RUL;
%RUL_avg = RUL_avg + RUL;
end
RUL_sensor=RUL_sensor/14;
engine_RUL = [engine_RUL ; RUL_sensor];
FD001_RUL = FD001_RUL + RUL_sensor;
FD001_RUL_all(g) = RUL_sensor;
RUL_trend = [RUL_trend ; RUL_sensor];  % plot RUL trends of 100 engines
%plot(RUL_sensor)
%title(RUL_sensor)
%pause(1)
end
FD001_RUL = FD001_RUL/100
% averageing
%stem(RUL_trend,'filled')
%title("Average RUL:  " + FD001_RUL)
%ylabel("RUL")
%xlabel("Engine No")
%figure
t = 1:100;
histogram(FD001_RUL_all)
title(min(FD001_RUL_all))
min(FD001_RUL_all)

%% Remove Features with Constant Values 
% m = min([XTrain{:}],[],2);
% M = max([XTrain{:}],[],2);
% idxConstant = M == m;
% 
% for i = 1:numel(XTrain)
%     XTrain{i}(idxConstant,:) = [];
% end
% 
% numFeatures = size(XTrain{1},1);

%% Filter and Normalize Training Predictors
  % Moving Median Filter
% SmoothData = XTrain;
% for i = 1:numel(SmoothData)
%     SmoothData{i} = SmoothData{i}'; % Transpose
% end
% for i = 1:numel(SmoothData)
%     SmoothData{i} = smoothdata(SmoothData{i},'movmedian');   % Moving Average Filter  'sgolay'= filter,'movmedian'
% end
%                                                                                        %    'rlowess'
% for i = 1:numel(SmoothData)
%     SmoothData{i} = SmoothData{i}'; % Transpose
% end
% XTrain = SmoothData;

%% min max normalization

% minVal = min([XTrain{:}],[],2); 
% maxVal = max([XTrain{:}],[],2);
% for i = 1:numel(XTrain)
%     XTrain{i} = (XTrain{i} - minVal) ./ ( maxVal - minVal );
% end
%norm_data = (bla - minVal) / ( maxVal - minVal );
%your_original_data = minVal + norm_data.*(maxVal - minVal);

%%
%mu = mean([XTrain{:}],2);
%sig = std([XTrain{:}],0,2);

%for i = 1:numel(XTrain)
%    XTrain{i} = (XTrain{i} - mu) ./ sig;
%end
% applying initial RUL value to all the target label in dataset which we
% have obtained from piece wise linear function
%YTrain_T = YTrain;
%YTrain_T = YTrain_T(engine_RUL);
thr = 87; % for rul limit to 150
for i = 1:numel(YTrain)
    %YTrain{i}(YTrain{i} > engine_RUL(i)) = engine_RUL(i); %% Threshold varies for each engine
    YTrain{i}(YTrain{i} > thr) = thr;
    %YTrain{i}(YTrain{i} > thr) = thr;
end
subplot(211)
plot(XTrain{1}')
title("Training Data, First Observation")
subplot(212)
plot(YTrain{1}')
title("Labels") % Degradation start from 150

%% Prepare Data for Padding
% To minimize the amount of padding added to the mini-batches, sort the training 
% data by sequence length. Then, choose a mini-batch size which divides the training 
% data evenly and reduces the amount of padding in the mini-batches.
% Sort the training data by sequence length.
clear cc
for i=1:numel(XTrain)
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

% split dataset into train & validation set (train= 90% & validation = 10%)
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
% figure
% bar(sequenceLengths)
% xlabel("Sequence")
% ylabel("Length")
% title("Sorted Data")

 %% Define LSTM Network Architecture
 % Define the network architecture. Create an LSTM network that consists of an LSTM layer
 % with 200 hidden units, followed by a fully connected layer of size 50 and a dropout layer
 % with dropout probability 0.5.
numResponses = size(YTrain{1},1);
numFeatures = size(XTrain{1},1);
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(60,'OutputMode','sequence')
    dropoutLayer(0.1)
    lstmLayer(60,'OutputMode','sequence')
    dropoutLayer(0.1)
    lstmLayer(60,'OutputMode','sequence')
     dropoutLayer(0.1)
     lstmLayer(60,'OutputMode','sequence')
     dropoutLayer(0.1)
%     lstmLayer(90,'OutputMode','sequence')
%     dropoutLayer(0.1)
%     lstmLayer(90,'OutputMode','sequence')
%     dropoutLayer(0.1)
    fullyConnectedLayer(70)
    dropoutLayer(0.5)
    fullyConnectedLayer(70)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];
% To prevent the gradients from exploding, set the gradient threshold to 1. 
% To keep the sequences sorted by length, set 'Shuffle' to 'never'.
maxEpochs = 150;
miniBatchSize = 15;
% SGD = sgdm,      adam,      rmsprop
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.02, ...
    'ValidationData',{valx,valy},...
    'ValidationFrequency',20,...
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
net = trainNetwork(XTrain,YTrain,layers,options);
%% Save the Trained Model
FD001_model = net
save ('FD003_m2w12th0.3.mat', 'FD001_model')

%% Load the Trained LSTM Model
load FD003_m2w8th0.1.mat;

%% Test The Network
filenamePredictors = fullfile(dataFolder,"test_FD004.txt");
filenameResponses = fullfile(dataFolder,"RUL_FD004.txt");
[XTest,YTest] = processTurboFanDataTest(filenamePredictors,filenameResponses);

for i = 1:numel(XTest)
    XTest{i}(idxConstant,:) = [];
end
% Remove constant values and clipping
SmoothData_Test = XTest;
for i = 1:numel(SmoothData_Test)
    SmoothData_Test{i} = SmoothData_Test{i}'; % Transpose
end
for i = 1:numel(SmoothData_Test)
    SmoothData_Test{i} = smoothdata(SmoothData_Test{i},'movmedian');   % Moving Average Filter
end

for i = 1:numel(SmoothData_Test)
    SmoothData_Test{i} = SmoothData_Test{i}'; % Transpose
end
XTest = SmoothData_Test;


% minVal = min([XTest{:}],[],2); 
% maxVal = max([XTest{:}],[],2);
% for i = 1:numel(XTest)
%     XTest{i} = (XTest{i} - minVal) ./ ( maxVal - minVal );
% end

for i = 1:numel(XTest)
    XTest{i} = (XTest{i} - mu) ./ sig;
    %YTest{i}(YTest{i} > engine_RUL(i)) = engine_RUL(i); %% Threshold varies for each engine
    YTest{i}(YTest{i} > thr) = thr;
end

%mu = mean([XTest{:}],2);
%sig = std([XTest{:}],0,2);



%% Prediction on Test Data
% for i=1:numel(XTest)
%     sequence = XTest{i};
%     sequenceLengthss(i) = size(sequence,2);
% end
% 
% [sequenceLengthss,idx] = sort(sequenceLengthss,'descend');
%XTest = XTest(idx);
%YTest = YTest(idx);
YPred = predict(FD001_model,XTest,'MiniBatchSize',1);
% YPred = [];
%  
% for i = 1:numel(XTest)
%     for j=1:size(XTest{i},2) 
%     [net,YPred{i}(:,j)] = predictAndUpdateState(net,XTest{i}(:,j),'ExecutionEnvironment','cpu');
%     end
% end

%YPred = sig*YPred + mu;
%idx = randperm(numel(YPred),4);
idx = [15 73 155 202]
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    
    plot(YTest{idx(i)},'--')
    hold on
    plot(YPred{idx(i)},'.-')
    hold off
    
    ylim([0 thr + 40])
    rmse = sqrt(mean((YPred{idx(i)} - YTest{idx(i)}).^2))
    title("RMSE = " + rmse)
    xlabel("Time Step")
    ylabel("RUL")
    legend(["Test Data" "Predicted"],'Location','northeast')
end


%% RMS Error & Score Function for RUL Prediction

for i = 1:numel(YTest)
      YTestLast(i) = YTest{i}(end);
      YPredLast(i) = YPred{i}(end);
  end
figure
subplot(224)
rmse = sqrt(mean((YPredLast - YTestLast).^2))
histogram(YPredLast - YTestLast)
title("RMSE = " + rmse)
ylabel("Frequency")
xlabel("Error")
hold on
%%
%figure
subplot(412)
plot(YTestLast)
hold on
plot(YPredLast,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("RUL")
xlabel("Test Engine No")
hold on
%title("Forecast with Updates")
%%
% Score Function

earlyPenalty = 13;
latePenalty = 10;
sub = YPredLast - YTestLast;
S = zeros(size(sub));
%%%LATE prediction, pred > trueRUL, pred - trueRUL > 0
f = find(sub >= 0);
FNR = length(f) / length(YTestLast); % false negative rate
S(f) = exp(sub(f)/latePenalty)-1;
%%%%EARLY
f = find(sub < 0);
FPR = length(f) / length(YTestLast); % false positive rate
S(f) = exp(-sub(f)/earlyPenalty)-1;
SCORE=sum(S)
er=sqrt(sub.^2);

subplot(2,1,2)
stem(YPredLast - YTestLast,'filled')
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse + "   " + "Score = " + SCORE)

%% Degradation Levels (0 to 10)
min_RUL = 0;
max_RUL = 97;
a=0;
label= zeros(100,1);
range = max_RUL - min_RUL;
range = round(range/10);
for n = 1: range
    for o = 1: size(YPredLast,2)
       if YPredLast(o) < n*range 
           if YPredLast(o) > (n-1)*range % Data labelling
             label(o) = n
           end
       end
    end
end
YPredLast_new =[YPredLast(1,1:80)' label(1:80,1)];

%% Save Linear Regression Model

save ('LR_trainedModel.mat','LR_trainedModel')

%% Load Linear Regression Model

load LR_trainedModel;

%% Prediction

deg_level = LR_trainedModel.predictFcn(YPredLast(1,80:100)') 
%plot(YPredLast',deg_level,'r--b')
createfigure(YPredLast(1,80:100)',label(80:100,1),deg_level,YPredLast(1,80:100)',abs(deg_level-label(80:100,1)))

%% Box Plot
fd01_1 = (YPredLast - YTestLast)';
fd01_2 = (YPredLast - YTestLast)';
fd01_3 = (YPredLast - YTestLast)';
fd01_4 = (YPredLast - YTestLast)';
fd01_5 = (YPredLast - YTestLast)';
subplot(224)
boxplot([fd01_1,fd01_2,fd01_3,fd01_4,fd01_5],...
    'Labels', {'0.05','0.07','0.09','0.1','0.2'})
hold on
plot([24.9 24.9 23 23 19.2]) % rmse
title('Boxplot of prediction')
xlabel('Threshold of piece wise linear')
ylabel('RUL')








