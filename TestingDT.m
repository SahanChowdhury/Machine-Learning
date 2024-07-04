%Testing optimized decision tree
rng(1)
% Load the saved decision tree model
load('decisionTree_HP.mat');

%%

%loading test data
load('test_data.mat');

% Define feature columns
X = {'Year', 'Month', 'Day', 'precipitation', 'temp_max', 'temp_min', 'wind', 'temp_range', 'Winter', 'Summer', 'Spring', 'Autumn'};
Y = 'weather_labels';

XTest = testingData(:, X);
YTest = testingData.(Y);


%%

predictions_dtTrainHP = predict(decisionTree_HP, XTest);
%displaying first couple rows of predictions
head(predictions_dtTrainHP);
%Saving the training model predictions in a csv
%writematrix(predictions_dtTrainHP, 'predictions_dtTrain.csv');
%%

%Accuracy
%Summing all correct predictions by comparing to YTest (True values)
correctPredictions_dtTrainHP = sum(YTest == predictions_dtTrainHP);
%Total number 
totalPredictions_dtTrainHP = length(YTest);

testAccuracy_dtTrainHP = correctPredictions_dtTrainHP /292;
AccuracyPercentage_dtTrainHP = testAccuracy_dtTrainHP *100 
disp(['Test Accuracy: ' num2str(testAccuracy_dtTrainHP)]);

%%
%Results
%Where dtTrainHP stands for Decision tree Training Hyper Parameter
results_dtTrainHP = confusionmat(YTest, predictions_dtTrainHP);
results_dtTrainHP
results_sum_dtTrainHP = sum(sum(results_dtTrainHP));
results_sum_dtTrainHP

figure;
dtTrain_HeatmapHP= heatmap(results_dtTrainHP);

%%
%Evaluation Metrics

%Drizzle
%Where dtTrainHP stands for Decision tree Training Hyper Parameter
% True Positive
TP_Class1_dtTrainHP = results_dtTrainHP(1, 1);

% False Negative
FN_Class1_dtTrainHP = sum(results_dtTrainHP(:, 1)) - TP_Class1_dtTrainHP;

% False Positive
FP_Class1_dtTrainHP = sum(results_dtTrainHP(1, :)) - TP_Class1_dtTrainHP;

% True Negative
TN_Class1_dtTrainHP = sum(results_dtTrainHP(:)) - (TP_Class1_dtTrainHP + FP_Class1_dtTrainHP + FN_Class1_dtTrainHP);

% Precision
precision_Class1_dtTrainHP = TP_Class1_dtTrainHP / (TP_Class1_dtTrainHP + FP_Class1_dtTrainHP);

% Recall (Sensitivity)
recall_Class1_dtTrainHP = TP_Class1_dtTrainHP / (TP_Class1_dtTrainHP + FN_Class1_dtTrainHP);

% F1 Score
f1Score_Class1_dtTrainHP = 2 * (precision_Class1_dtTrainHP * recall_Class1_dtTrainHP) / (precision_Class1_dtTrainHP + recall_Class1_dtTrainHP);

% Accuracy
accuracy_Class1_dtTrainHP = (TP_Class1_dtTrainHP + TN_Class1_dtTrainHP) / sum(results_dtTrainHP(:));

disp(['Class 1 (Drizzle)']);
disp(['True Positive: ', num2str(TP_Class1_dtTrainHP)]);
disp(['False Negative: ', num2str(FN_Class1_dtTrainHP)]);
disp(['False Positive: ', num2str(FP_Class1_dtTrainHP)]);
disp(['True Negative: ', num2str(TN_Class1_dtTrainHP)]);
disp(['Precision: ', num2str(precision_Class1_dtTrainHP)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class1_dtTrainHP)]);
disp(['F1 Score: ', num2str(f1Score_Class1_dtTrainHP)]);
disp(['Accuracy: ', num2str(accuracy_Class1_dtTrainHP)]);
disp('----------------------');

%------------------------------------------
%Fog

TP_Class2_dtTrainHP = results_dtTrainHP(2, 2);
FN_Class2_dtTrainHP = sum(results_dtTrainHP(:, 2)) - TP_Class2_dtTrainHP;
FP_Class2_dtTrainHP = sum(results_dtTrainHP(2, :)) - TP_Class2_dtTrainHP;
TN_Class2_dtTrainHP = sum(results_dtTrainHP(:)) - (TP_Class2_dtTrainHP + FP_Class2_dtTrainHP + FN_Class2_dtTrainHP);
precision_Class2_dtTrainHP = TP_Class2_dtTrainHP / (TP_Class2_dtTrainHP + FP_Class2_dtTrainHP);
recall_Class2_dtTrainHP = TP_Class2_dtTrainHP / (TP_Class2_dtTrainHP + FN_Class2_dtTrainHP);
f1Score_Class2_dtTrainHP = 2 * (precision_Class2_dtTrainHP * recall_Class2_dtTrainHP) / (precision_Class2_dtTrainHP + recall_Class2_dtTrainHP);
accuracy_Class2_dtTrainHP = (TP_Class2_dtTrainHP + TN_Class2_dtTrainHP) / sum(results_dtTrainHP(:));

disp('Class 2 (Fog)');
disp(['True Positive: ', num2str(TP_Class2_dtTrainHP)]);
disp(['False Negative: ', num2str(FN_Class2_dtTrainHP)]);
disp(['False Positive: ', num2str(FP_Class2_dtTrainHP)]);
disp(['True Negative: ', num2str(TN_Class2_dtTrainHP)]);
disp(['Precision: ', num2str(precision_Class2_dtTrainHP)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class2_dtTrainHP)]);
disp(['F1 Score: ', num2str(f1Score_Class2_dtTrainHP)]);
disp(['Accuracy: ', num2str(accuracy_Class2_dtTrainHP)]);
disp('----------------------');
%--------------------------------------------------

%Rain 
TP_Class3_dtTrainHP = results_dtTrainHP(3, 3);
FN_Class3_dtTrainHP = sum(results_dtTrainHP(:, 3)) - TP_Class3_dtTrainHP;
FP_Class3_dtTrainHP = sum(results_dtTrainHP(3, :)) - TP_Class3_dtTrainHP;
TN_Class3_dtTrainHP = sum(results_dtTrainHP(:)) - (TP_Class3_dtTrainHP + FP_Class3_dtTrainHP + FN_Class3_dtTrainHP);
precision_Class3_dtTrainHP = TP_Class3_dtTrainHP / (TP_Class3_dtTrainHP + FP_Class3_dtTrainHP);
recall_Class3_dtTrainHP = TP_Class3_dtTrainHP / (TP_Class3_dtTrainHP + FN_Class3_dtTrainHP);
f1Score_Class3_dtTrainHP = 2 * (precision_Class3_dtTrainHP * recall_Class3_dtTrainHP) / (precision_Class3_dtTrainHP + recall_Class3_dtTrainHP);
accuracy_Class3_dtTrainHP = (TP_Class3_dtTrainHP + TN_Class3_dtTrainHP) / sum(results_dtTrainHP(:));

disp('Class 3 (Rain)');
disp(['True Positive: ', num2str(TP_Class3_dtTrainHP)]);
disp(['False Negative: ', num2str(FN_Class3_dtTrainHP)]);
disp(['False Positive: ', num2str(FP_Class3_dtTrainHP)]);
disp(['True Negative: ', num2str(TN_Class3_dtTrainHP)]);
disp(['Precision: ', num2str(precision_Class3_dtTrainHP)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class3_dtTrainHP)]);
disp(['F1 Score: ', num2str(f1Score_Class3_dtTrainHP)]);
disp(['Accuracy: ', num2str(accuracy_Class3_dtTrainHP)]);
disp('----------------------');

%--------------------------------------------------
%Snow
TP_Class4_dtTrainHP = results_dtTrainHP(4, 4);
FN_Class4_dtTrainHP = sum(results_dtTrainHP(:, 4)) - TP_Class4_dtTrainHP;
FP_Class4_dtTrainHP = sum(results_dtTrainHP(4, :)) - TP_Class4_dtTrainHP;
TN_Class4_dtTrainHP = sum(results_dtTrainHP(:)) - (TP_Class4_dtTrainHP + FP_Class4_dtTrainHP + FN_Class4_dtTrainHP);
precision_Class4_dtTrainHP = TP_Class4_dtTrainHP / (TP_Class4_dtTrainHP + FP_Class4_dtTrainHP);
recall_Class4_dtTrainHP = TP_Class4_dtTrainHP / (TP_Class4_dtTrainHP + FN_Class4_dtTrainHP);
f1Score_Class4_dtTrainHP = 2 * (precision_Class4_dtTrainHP * recall_Class4_dtTrainHP) / (precision_Class4_dtTrainHP + recall_Class4_dtTrainHP);
accuracy_Class4_dtTrainHP = (TP_Class4_dtTrainHP + TN_Class4_dtTrainHP) / sum(results_dtTrainHP(:));

disp('Class 4 (Snow)');
disp(['True Positive: ', num2str(TP_Class4_dtTrainHP)]);
disp(['False Negative: ', num2str(FN_Class4_dtTrainHP)]);
disp(['False Positive: ', num2str(FP_Class4_dtTrainHP)]);
disp(['True Negative: ', num2str(TN_Class4_dtTrainHP)]);
disp(['Precision: ', num2str(precision_Class4_dtTrainHP)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class4_dtTrainHP)]);
disp(['F1 Score: ', num2str(f1Score_Class4_dtTrainHP)]);
disp(['Accuracy: ', num2str(accuracy_Class4_dtTrainHP)]);
disp('----------------------');


%--------------------------------------------------
%Sun
TP_Class5_dtTrainHP = results_dtTrainHP(5, 5);
FN_Class5_dtTrainHP = sum(results_dtTrainHP(:, 5)) - TP_Class5_dtTrainHP;
FP_Class5_dtTrainHP = sum(results_dtTrainHP(5, :)) - TP_Class5_dtTrainHP;
TN_Class5_dtTrainHP = sum(results_dtTrainHP(:)) - (TP_Class5_dtTrainHP + FP_Class5_dtTrainHP + FN_Class5_dtTrainHP);
precision_Class5_dtTrainHP = TP_Class5_dtTrainHP / (TP_Class5_dtTrainHP + FP_Class5_dtTrainHP);
recall_Class5_dtTrainHP = TP_Class5_dtTrainHP / (TP_Class5_dtTrainHP + FN_Class5_dtTrainHP);
f1Score_Class5_dtTrainHP = 2 * (precision_Class5_dtTrainHP * recall_Class5_dtTrainHP) / (precision_Class5_dtTrainHP + recall_Class5_dtTrainHP);
accuracy_Class5_dtTrainHP = (TP_Class5_dtTrainHP + TN_Class5_dtTrainHP) / sum(results_dtTrainHP(:));
disp(['Class 5 (Sun)']);
disp(['True Positive: ', num2str(TP_Class5_dtTrainHP)]);
disp(['False Negative: ', num2str(FN_Class5_dtTrainHP)]);
disp(['False Positive: ', num2str(FP_Class5_dtTrainHP)]);
disp(['True Negative: ', num2str(TN_Class5_dtTrainHP)]);
disp(['Precision: ', num2str(precision_Class5_dtTrainHP)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class5_dtTrainHP)]);
disp(['F1 Score: ', num2str(f1Score_Class5_dtTrainHP)]);
disp(['Accuracy: ', num2str(accuracy_Class5_dtTrainHP)]);
disp('----------------------');

%--------------------------------------------------

%%
