%% Create a directory to store the Turbofan Engine Degradation Simulation data set.
clc, close all, clear all
dataFolder = fullfile('D:\MS EE\Predictive Maintenance\Datasets',"Challenge_Data");
if ~exist(dataFolder,'dir')
    mkdir(dataFolder);
end
%filename = "CMAPSSData.zip";
%unzip(filename,dataFolder)

%% Prepare Training Data             %% (STEP # 1)
filenamePredictors = fullfile(dataFolder,"train.txt");    
[XTrain,YTrain] = processTurboFanDataTrain(filenamePredictors);  % split data into train & test

%% Correltaion Analysis          %% (STEP # 2)
FD001_cc_d =zeros(24,1);
engines = 218;       % No of engines  , FD001=100, FD002=260, FD003=100, Fd004=249
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
% FD001   These threshold are also given in paper. sensor selection
% criteria
% FD001 = < 10,   FD002 = < 5, FD003= < 10, FD004 = < 5 
idxConstant = FD001_cc_d < 5;   
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
window = 10; % it is given in experimental section.
e=0;
RUL_trend = [];
RUL_avg = [];
RUL_sensor=0;
FD001_RUL=0;
FD001_RUL_all = [];
for g=1:218  % engines FD001 = 100, FD002 = 260, FD003 = 100, FD004 = 249
 
% This alorithm is given in paper    
for f = 1:24  % sensor  FD001 = 14, FD002 = 24 , FD003 = 16 , FD004 = 24
    w1 = XTrain{g}(f, 1:window);
    w1= abs(w1);
for i= 1:floor(numel(YTrain{g})/window)-1 
    w2 = XTrain{g}(f , ((window)*i)+1:window*(i+1));
    w2= abs(w2);
    mean_1 = (mean(w1));
    mean_2 = (mean(w2));
    diff = abs(mean_1 - mean_2);
    RUL(i)= diff;
    if (diff >=0.5 && e==0)
       initial_RUL =numel(YTrain{g}) - window*i;  % RUL Calculation
       e=1;
    end    
end
e=0;
RUL_sensor = RUL_sensor + initial_RUL;
%RUL_avg = RUL_avg + RUL;
end
RUL_sensor=RUL_sensor/24;
FD001_RUL = FD001_RUL + RUL_sensor;
FD001_RUL_all(g) = RUL_sensor;
RUL_trend = [RUL_trend ; RUL_sensor];  % plot RUL trends of 100 engines
%plot(RUL_sensor)
%title(RUL_sensor)
%pause(1)
end
FD001_RUL = FD001_RUL/218   % averageing
stem(RUL_trend,'filled')
title("Average RUL:  " + FD001_RUL)
ylabel("RUL")
xlabel("Engine No")
figure
t = 1:100;
histogram(FD001_RUL_all)
title(min(FD001_RUL_all))


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

thr = 125; % for rul limit to 150
for i = 1:numel(YTrain)
    YTrain{i}(YTrain{i} > thr) = thr;
end
plot(XTrain{2}')
title("Training Data, First Observation")
figure
plot(YTrain{1}')
title("Labels") % Degradation start from 150

%% Data Preparation for Convolution Neural Network
% for i=1:numel(XTrain)
%     sequence = XTrain{i};
%     sequenceLengths(i) = size(sequence,2);
% end
% 
% % split dataset into train & validation set (train= 90% & validation = 10%)
% [sequenceLengths,idx] = sort(sequenceLengths,'descend');
% XTrain = XTrain(idx);
% YTrain = YTrain(idx);
% XTrain_data =[];
% YTrain_label =[];
% % WINDOW SIZE = 5
% for d = 1 : 218
% data_s = size (XTrain{d},2);
% dim = rem(data_s,5);
% 
%    for xsam=1:(data_s-dim)/5
%      XSam{xsam} = XTrain{d}(:,(5*xsam)-4:5*xsam);
%      YSam{xsam} = YTrain{d}((xsam*5));
%    end
% %xsam = xsam + (data_s-dim)/5;   
% %XSam=XSam';
% XTrain_data = [XTrain_data  XSam];
% YTrain_label =[YTrain_label YSam];
% clear XSam
% clear YSam
% end
% XTrain_data = XTrain_data';
% YTrain_label =YTrain_label';
%% Prepare Data for Padding
% To minimize the amount of padding added to the mini-batches, sort the training 
% data by sequence length. Then, choose a mini-batch size which divides the training 
% data evenly and reduces the amount of padding in the mini-batches.
% Sort the training data by sequence length.
for i=1:numel(XTrain)
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

% split dataset into train & validation set (train= 90% & validation = 10%)
[sequenceLengths,idx] = sort(sequenceLengths,'descend');
XTrain = XTrain(idx);
YTrain = YTrain(idx);
for  i =1:196                     % 234=FD002, 90=FD001,FD003, 224=FD004
    bb{i}=XTrain{i};
    ccc{i}=YTrain{i};
end
for k= 1:22                     % 26=FD002,    10=FD001,FD003, 25=FD004
    j=k+196;                        
    valx{k}=XTrain{j};
    valy{k}=YTrain{j};  
end
XTrain=bb';
YTrain=ccc';
valx=valx';
valy=valy';
figure
bar(sequenceLengths)
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")


 %% Define GRU - LSTM Network Architecture
 % Define the network architecture. Create an LSTM network that consists of an LSTM layer
 % with 200 hidden units, followed by a fully connected layer of size 50 and a dropout layer
 % with dropout probability 0.5.
numResponses = size(YTrain{1},1);
numFeatures = size(XTrain{1},1);
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures,'Name','input')
    lstmLayer(7,'OutputMode','sequence','Name','lstm-1')
    dropoutLayer(0.2,'Name','do-1')
    %additionLayer(2,'Name','add21')
    lstmLayer(7,'OutputMode','sequence','Name','lstm-2')
    dropoutLayer(0.2,'Name','do-21')
    additionLayer(2,'Name','add22')
    gruLayer(7,'OutputMode','sequence','Name','lstm-71')
    %reluLayer('Name','relu-9')
    dropoutLayer(0.2,'Name','do-23')
    additionLayer(2,'Name','add23')
    %additionLayer(2,'Name','add11')
    gruLayer(7,'OutputMode','sequence','Name','lstm-72')
    dropoutLayer(0.2,'Name','do-24')
    additionLayer(3,'Name','add24')
    gruLayer(7,'OutputMode','sequence','Name','lstm-73')
    dropoutLayer(0.2,'Name','do-25')
    additionLayer(3,'Name','add26')
    gruLayer(7,'OutputMode','sequence','Name','lstm-74')
    dropoutLayer(0.2,'Name','do-26')
    additionLayer(2,'Name','add25')
    %gruLayer(7,'OutputMode','sequence','Name','lstm-73')
    %dropoutLayer(0.2,'Name','do-25')
    %additionLayer(2,'Name','add12')

    %additionLayer(2,'Name','add13')
    additionLayer(6,'Name','concat')
    %gruLayer(1,'OutputMode','sequence','Name','lstm-5')
    %dropoutLayer(0.2,'Name','do-4')
    %gruLayer(1,'OutputMode','sequence','Name','lstm-10')
    %dropoutLayer(0.2,'Name','do-41')
    %gruLayer(14,'OutputMode','sequence','Name','lstm-12')
    %dropoutLayer(0.2,'Name','do-42')
    %reluLayer('Name','relu-8')
    %gruLayer(14,'OutputMode','sequence','Name','lstm-13')
    %dropoutLayer(0.2,'Name','do-43')
    fullyConnectedLayer(80,'Name','fc-1')
    reluLayer('Name','relu-1')
    dropoutLayer(0.5,'Name','do-5')
    fullyConnectedLayer(80,'Name','fc-2')
    reluLayer('Name','relu-2')
    dropoutLayer(0.5,'Name','do-6')
    fullyConnectedLayer(numResponses,'Name','fc-3')
    regressionLayer('Name','reg_out')];
lgraph = layerGraph(layers);
%fold = sequenceFoldingLayer('Name','fold');
%unfold = sequenceUnfoldingLayer('Name','unfold');
%flatten = flattenLayer('Name','flatten');
drop1 = dropoutLayer(0.2,'Name','drop1');
drop2 = dropoutLayer(0.2,'Name','drop2');
drop3 = dropoutLayer(0.2,'Name','drop3');
drop4 = dropoutLayer(0.2,'Name','drop4');
drop5 = dropoutLayer(0.2,'Name','drop5');
drop6 = dropoutLayer(0.2,'Name','drop6');
drop7 = dropoutLayer(0.2,'Name','drop7');
drop8 = dropoutLayer(0.2,'Name','drop8');

add13 = concatenationLayer(1,2,'Name','add13');
add14 = concatenationLayer(1,2,'Name','add14');
add15 = concatenationLayer(1,2,'Name','add15');
add16 = concatenationLayer(1,2,'Name','add16');
add17 = additionLayer(2,'Name','add17');
skipConv =  lstmLayer(7,'OutputMode','sequence','Name','skipConv');
skipConv2 = gruLayer(7,'OutputMode','sequence','Name','skipConv2');
skipConv3 = lstmLayer(7,'OutputMode','sequence','Name','skipConv3');
skipConv4 = gruLayer(7,'OutputMode','sequence','Name','skipConv4');
skipConv5 = gruLayer(7,'OutputMode','sequence','Name','skipConv5');
skipConv6 = gruLayer(7,'OutputMode','sequence','Name','skipConv6');
skipConv7 = gruLayer(7,'OutputMode','sequence','Name','skipConv7');
skipConv8 = gruLayer(7,'OutputMode','sequence','Name','skipConv8');
skipConv9 = gruLayer(7,'OutputMode','sequence','Name','skipConv9');
skipConv10 = gruLayer(7,'OutputMode','sequence','Name','skipConv10');
skipConv11 = gruLayer(7,'OutputMode','sequence','Name','skipConv11');
%  lgraph = addLayers(lgraph,skipConv);
%  lgraph = addLayers(lgraph,skipConv2);
%  lgraph = addLayers(lgraph,skipConv3);
% % lgraph = addLayers(lgraph,skipConv4);
%  lgraph = addLayers(lgraph,skipConv5);
%  lgraph = addLayers(lgraph,skipConv6);
%  lgraph = addLayers(lgraph,skipConv7);
% % lgraph = addLayers(lgraph,skipConv8);
% % lgraph = addLayers(lgraph,skipConv9);
% % lgraph = addLayers(lgraph,skipConv10);
% % lgraph = addLayers(lgraph,skipConv11);
% lgraph = addLayers(lgraph,add13);
% lgraph = addLayers(lgraph,add14);
% lgraph = addLayers(lgraph,add15);
% %lgraph = addLayers(lgraph,add16);
% %lgraph = addLayers(lgraph,add17);
% % = addLayers(lgraph,fold);
%  lgraph = addLayers(lgraph,drop1);
%  lgraph = addLayers(lgraph,drop2);
%  lgraph = addLayers(lgraph,drop3);
% % lgraph = addLayers(lgraph,drop4);
%  lgraph = addLayers(lgraph,drop5);
%  lgraph = addLayers(lgraph,drop6);
%  lgraph = addLayers(lgraph,drop7);
% lgraph = addLayers(lgraph,drop8);

%lgraph = addLayers(lgraph,unfold);
%lgraph = addLayers(lgraph,flatten);
%lgraph = addLayers(lgraph,relu);

%lgraph = connectLayers(lgraph,'lstm-1','add21/in2');
lgraph = connectLayers(lgraph,'do-1','add22/in2');
lgraph = connectLayers(lgraph,'do-1','add24/in3');
lgraph = connectLayers(lgraph,'do-21','add23/in2');
lgraph = connectLayers(lgraph, 'do-23', 'add24/in2');
lgraph = connectLayers(lgraph, 'do-23', 'add26/in3');
lgraph = connectLayers(lgraph, 'do-25', 'add25/in2');
lgraph = connectLayers(lgraph, 'do-24', 'add26/in2');
lgraph = connectLayers(lgraph,'do-24','concat/in5');
lgraph = connectLayers(lgraph,'do-25','concat/in6');
lgraph = connectLayers(lgraph,'do-1','concat/in2');
lgraph = connectLayers(lgraph,'do-21','concat/in3');
lgraph = connectLayers(lgraph,'do-23','concat/in4');
%lgraph = connectLayers(lgraph,'add24/out','concat/in6');

%lgraph = connectLayers(lgraph,'input','fold');
% lgraph = connectLayers(lgraph,'input','skipConv');
% lgraph = connectLayers(lgraph,'skipConv','drop1');
% %lgraph = connectLayers(lgraph, 'drop1', 'concat/in2');
% lgraph = connectLayers(lgraph,'input','skipConv2');
% lgraph = connectLayers(lgraph,'skipConv2','drop2');
% %lgraph = connectLayers(lgraph, 'drop2', 'concat/in3');
% lgraph = connectLayers(lgraph,'input','skipConv3');
% lgraph = connectLayers(lgraph,'skipConv3','drop3');
% %lgraph = connectLayers(lgraph,'drop3', 'concat/in4');
% %lgraph = connectLayers(lgraph,'input','skipConv4');
% %lgraph = connectLayers(lgraph,'skipConv4','drop4');
% %lgraph = connectLayers(lgraph, 'drop4', 'concat/in5');
% lgraph = connectLayers(lgraph, 'add13/out', 'skipConv5');
% lgraph = connectLayers(lgraph, 'skipConv5', 'drop5');
% lgraph = connectLayers(lgraph, 'drop5', 'concat/in2');
% lgraph = connectLayers(lgraph, 'drop1', 'add13/in1');
% lgraph = connectLayers(lgraph, 'do-1', 'add13/in2');
% lgraph = connectLayers(lgraph, 'drop1', 'add14/in1');
% lgraph = connectLayers(lgraph, 'drop2', 'add14/in2');
% lgraph = connectLayers(lgraph, 'add14/out', 'skipConv6');
% lgraph = connectLayers(lgraph, 'skipConv6', 'drop6');
% lgraph = connectLayers(lgraph, 'drop6', 'concat/in3');
% lgraph = connectLayers(lgraph, 'drop2', 'add15/in1');
% lgraph = connectLayers(lgraph, 'drop3', 'add15/in2');
% lgraph = connectLayers(lgraph, 'add15/out', 'skipConv7');
% lgraph = connectLayers(lgraph, 'skipConv7', 'drop7');
% lgraph = connectLayers(lgraph, 'drop7', 'concat/in4');
% %lgraph = connectLayers(lgraph, 'drop3', 'add16/in1');
%lgraph = connectLayers(lgraph, 'drop4', 'add16/in2');
%lgraph = connectLayers(lgraph, 'add16/out', 'skipConv8');
%lgraph = connectLayers(lgraph, 'skipConv8', 'drop8');
%lgraph = connectLayers(lgraph, 'drop8', 'concat/in5');
% lgraph = connectLayers(lgraph, 'lstm-1', 'concat/in6');
% lgraph = connectLayers(lgraph, 'skipConv', 'concat/in7');
% lgraph = connectLayers(lgraph, 'skipConv2', 'concat/in8');
% lgraph = connectLayers(lgraph, 'skipConv3', 'concat/in9');
% lgraph = connectLayers(lgraph, 'skipConv4', 'concat/in10');
%lgraph = connectLayers(lgraph, 'lstm-2', 'add16/in1');
% lgraph = connectLayers(lgraph, 'lstm-72', 'add17/in2');
% lgraph = connectLayers(lgraph, 'skipConv4', 'add17/in1');
% lgraph = connectLayers(lgraph, 'input', 'skipConv9');
% lgraph = connectLayers(lgraph, 'input', 'skipConv10');
% lgraph = connectLayers(lgraph, 'input', 'skipConv11');
% lgraph = connectLayers(lgraph, 'skipConv2', 'add16/in2');
%lgraph = connectLayers(lgraph, 'add16/out','skipConv6');
%lgraph = connectLayers(lgraph, 'skipConv6','skipConv7');
%lgraph = connectLayers(lgraph, 'skipConv7','drop5');
%lgraph = connectLayers(lgraph, 'skipConv5','skipConv8');
%lgraph = connectLayers(lgraph, 'skipConv8','drop6');

% lgraph = connectLayers(lgraph, 'lstm-71', 'add15/in1');
% lgraph = connectLayers(lgraph, 'skipConv3', 'add15/in2');
% lgraph = connectLayers(lgraph, 'add15/out', 'skipConv5');
% lgraph = connectLayers(lgraph, 'drop5', 'concat/in3');
% lgraph = connectLayers(lgraph, 'drop6', 'concat/in4');
% lgraph = connectLayers(lgraph, 'add17/out', 'concat/in5');
% lgraph = connectLayers(lgraph, 'skipConv9', 'concat/in6');
% lgraph = connectLayers(lgraph, 'skipConv10', 'concat/in7');
% lgraph = connectLayers(lgraph, 'skipConv11', 'concat/in8');

%lgraph = connectLayers(lgraph, 'drop2', 'add11/in2');

%lgraph = connectLayers(lgraph,'max-3','unfold/in');
%lgraph = connectLayers(lgraph,'unfold/out','flatten');
%lgraph = connectLayers(lgraph, 'flatten', 'concat/in2');
%lgraph = connectLayers(lgraph,'skipConv','add12/in2');
%skiplstm = gruLayer(14,'OutputMode','sequence','Name','skiplstm');
%lgraph = addLayers(lgraph,skiplstm);
%lgraph = connectLayers(lgraph,'lstm-1','add12/in2');
%lgraph = connectLayers(lgraph,'flatten','add13/in3');
%lgraph = connectLayers(lgraph,'do-24','add13/in2');
%lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');
%lgraph = connectLayers(lgraph,'do-1','skiplstm');
%lgraph = connectLayers(lgraph,'do-1','add11/in2');
%lgraph = connectLayers(lgraph,'do-4','add14/in2');
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
plot(lgraph);
% To prevent the gradients from exploding, set the gradient threshold to 1. 
% To keep the sequences sorted by length, set 'Shuffle' to 'never'.
maxEpochs = 160;
miniBatchSize = 28;
% SGD = sgdm,      adam,      rmsprop
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',80, ...
    'LearnRateDropFactor',0.2, ...
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

net = trainNetwork(XTrain,YTrain,lgraph,options);
%% Save the Trained Model
model_new = net
save ('model_new.mat', 'model_new')

%% Load the Trained LSTM Model
load model_new;
net = model_new;

%% Test The Network
filenamePredictors = fullfile(dataFolder,"test.txt");
filenameResponses = fullfile(dataFolder,"RUL_FD001.txt");
[XTest,YTest] = processTurboFanDataTest(filenamePredictors,filenameResponses);

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
    %YTest{i}(YTest{i} > thr) = thr;
end

%mu = mean([XTest{:}],2);
%sig = std([XTest{:}],0,2);

% %%
% XTest_data =[];
% %YTest_label =[];
% % WINDOW SIZE = 5
% for d = 1 : 218
% data_s = size (XTest{d},2);
% dim = rem(data_s,5);
% 
%    for xsam=1:(data_s-dim)/5
%      XTSam{xsam} = XTest{d}(:,(5*xsam)-4:5*xsam);
%      %YTSam{xsam} = YTest{d}((xsam*5));
%    end
% %xsam = xsam + (data_s-dim)/5;   
% %XSam=XSam';
% XTest_data = [XTest_data  XTSam];
% %YTest_label =[YTest_label YTSam];
% clear XTSam
% %clear YTSam
% end
% XTest_data = XTest_data';
% %YTest_label =YTest_label';
% 
% % Prediction on Test Data

YPred = predict(net,XTest,'MiniBatchSize',28);
for i = 1:numel(XTest)
    %YTestLast(i) = YTest{i}(end);
    YPredLast(i) = YPred{i}(end);
end
plot(YPredLast)


%YPredLastAcc =[];
%sum =0;

%for i = 1:numel(XTest_data)
%     for d = 1:218
%     data_s = size (XTest{d},2);
%     dim = rem(data_s,5);
%     sum = sum + (data_s-dim)/5
%     
%     YPredLast = YPred(sum);
%      
%     YPredLastAcc = [YPredLastAcc  YPredLast];
%    end
%end
%YPredLastAcc = YPredLastAcc';

%A = cell2mat(YPredLastAcc);
%plot(A);

% YPred = [];
%  
% for i = 1:numel(XTest)
%     for j=1:size(XTest{i},2) 
%     [net,YPred{i}(:,j)] = predictAndUpdateState(net,XTest{i}(:,j),'ExecutionEnvironment','cpu');
%     end
% end

%YPred = sig*YPred + mu;
%idx = randperm(numel(YPred),4);

%% RMS Error & Score Function for RUL Prediction

% for i = 1:numel(YTest)
%     YTestLast(i) = YTest{i}(end);
%     %YPredLast(i) = YPred{i}(end);
% end
figure
subplot(224)
rmse = sqrt(mean((YPredLast - YTestLast).^2))
histogram(YPredLast - YTestLast)
title("RMSE = " + rmse)
ylabel("Frequency")
xlabel("Error")

figure
subplot(211)
plot(YTestLast)
hold on
plot(YPredLast,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("RUL")
%title("Forecast with Updates")

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
xlabel("Engine")
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
fd01_6 = (YPredLast - YTestLast)';
fd01_7 = (YPredLast - YTestLast)';
fd01_8 = (YPredLast - YTestLast)';
fd01_9 = (YPredLast - YTestLast)';
fd01_10 = (YPredLast - YTestLast)';
subplot(224)
boxplot([fd01_1,fd01_2,fd01_3,fd01_4,fd01_5,fd01_6,fd01_7,fd01_8,fd01_9,fd01_10],...
    'Labels', {'0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.10'})
hold on
plot([31.8 31.8 31.8 31.8 28 28 24.9 23.6 20.3 21.1 ]) % rmse
title('Boxplot of prediction')
xlabel('Threshold of piece wise linear')
ylabel('RUL')








