%Testing Optimized Random Forest Model
rng(1)
% Load the saved decision tree model
load('RandomForest_HP.mat');


%loading test data

load('test_data.mat');
%cv2 = crossval(decisiontreeHP, 'Holdout', 0.2);

% Define feature columns
X = {'Year', 'Month', 'Day', 'precipitation', 'temp_max', 'temp_min', 'wind', 'temp_range', 'Winter', 'Summer', 'Spring', 'Autumn'};
Y = 'weather_labels';

XTest = testingData(:, X);

YTest = testingData.(Y);


%Where rfTrainHP stands for Random Forest Training Hyper Parameter
predictions_rfTrainHP = predict(RandomForest_HP, XTest);
%displaying first couple rows of predictions
head(predictions_rfTrainHP);
%Saving the training model predictions in a csv
%writematrix(predictions_dtTrainHP, 'predictions_dtTrain.csv');


%Accuracy
%Summing all correct predictions by comparing to YTest (True values)
correctPredictions_rfTrainHP = sum(YTest == predictions_rfTrainHP);
%Total number 
totalPredictions_rfTrainHP = length(YTest);

% Calculate test accuracy
%By diving number of corect predictions by 
%Number of correct predictions/(lenght of test set = 292)
testAccuracy_rfTrainHP = correctPredictions_rfTrainHP /292;
AccuracyPercentage_rfTrainHP = testAccuracy_rfTrainHP *100 
disp(['Test Accuracy: ' num2str(testAccuracy_rfTrainHP)]);

%Results
%
results_rfTrainHP = confusionmat(YTest, predictions_rfTrainHP);
results_rfTrainHP
results_sum_rfTrainHP = sum(sum(results_rfTrainHP));
results_sum_rfTrainHP

figure;
rfTrain_HeatmapHP= heatmap(results_rfTrainHP);

%Drizzle
TP_Class1_rfTrainHP = results_rfTrainHP(1, 1);
FN_Class1_rfTrainHP = sum(results_rfTrainHP(:, 1)) - TP_Class1_rfTrainHP;
FP_Class1_rfTrainHP = sum(results_rfTrainHP(1, :)) - TP_Class1_rfTrainHP;
TN_Class1_rfTrainHP = sum(results_rfTrainHP(:)) - (TP_Class1_rfTrainHP + FP_Class1_rfTrainHP + FN_Class1_rfTrainHP);
precision_Class1_rfTrainHP = TP_Class1_rfTrainHP / (TP_Class1_rfTrainHP + FP_Class1_rfTrainHP);
recall_Class1_rfTrainHP = TP_Class1_rfTrainHP / (TP_Class1_rfTrainHP + FN_Class1_rfTrainHP);
f1Score_Class1_rfTrainHP = 2 * (precision_Class1_rfTrainHP * recall_Class1_rfTrainHP) / (precision_Class1_rfTrainHP + recall_Class1_rfTrainHP);
accuracy_Class1_rfTrainHP = (TP_Class1_rfTrainHP + TN_Class1_rfTrainHP) / sum(results_rfTrainHP(:));

disp(['Class 1 (Drizzle)']);
disp(['True Positive: ', num2str(TP_Class1_rfTrainHP)]);
disp(['False Negative: ', num2str(FN_Class1_rfTrainHP)]);
disp(['False Positive: ', num2str(FP_Class1_rfTrainHP)]);
disp(['True Negative: ', num2str(TN_Class1_rfTrainHP)]);
disp(['Precision: ', num2str(precision_Class1_rfTrainHP)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class1_rfTrainHP)]);
disp(['F1 Score: ', num2str(f1Score_Class1_rfTrainHP)]);
disp(['Accuracy: ', num2str(accuracy_Class1_rfTrainHP)]);
disp('----------------------');

%----------------------------------------------------
%Fog
TP_Class2_rfTrainHP = results_rfTrainHP(2, 2);
FN_Class2_rfTrainHP = sum(results_rfTrainHP(:, 2)) - TP_Class2_rfTrainHP;
FP_Class2_rfTrainHP = sum(results_rfTrainHP(2, :)) - TP_Class2_rfTrainHP;
TN_Class2_rfTrainHP = sum(results_rfTrainHP(:)) - (TP_Class2_rfTrainHP + FP_Class2_rfTrainHP + FN_Class2_rfTrainHP);
precision_Class2_rfTrainHP = TP_Class2_rfTrainHP / (TP_Class2_rfTrainHP + FP_Class2_rfTrainHP);
recall_Class2_rfTrainHP = TP_Class2_rfTrainHP / (TP_Class2_rfTrainHP + FN_Class2_rfTrainHP);
f1Score_Class2_rfTrainHP = 2 * (precision_Class2_rfTrainHP * recall_Class2_rfTrainHP) / (precision_Class2_rfTrainHP + recall_Class2_rfTrainHP);
accuracy_Class2_rfTrainHP = (TP_Class2_rfTrainHP + TN_Class2_rfTrainHP) / sum(results_rfTrainHP(:));

disp(['Class 2 (Fog)']);
disp(['True Positive: ', num2str(TP_Class2_rfTrainHP)]);
disp(['False Negative: ', num2str(FN_Class2_rfTrainHP)]);
disp(['False Positive: ', num2str(FP_Class2_rfTrainHP)]);
disp(['True Negative: ', num2str(TN_Class2_rfTrainHP)]);
disp(['Precision: ', num2str(precision_Class2_rfTrainHP)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class2_rfTrainHP)]);
disp(['F1 Score: ', num2str(f1Score_Class2_rfTrainHP)]);
disp(['Accuracy: ', num2str(accuracy_Class2_rfTrainHP)]);
disp('----------------------');
%----------------------------------------------------
%Rain
TP_Class3_rfTrainHP = results_rfTrainHP(3, 3);
FN_Class3_rfTrainHP = sum(results_rfTrainHP(:, 3)) - TP_Class3_rfTrainHP;
FP_Class3_rfTrainHP = sum(results_rfTrainHP(3, :)) - TP_Class3_rfTrainHP;
TN_Class3_rfTrainHP = sum(results_rfTrainHP(:)) - (TP_Class3_rfTrainHP + FP_Class3_rfTrainHP + FN_Class3_rfTrainHP);
precision_Class3_rfTrainHP = TP_Class3_rfTrainHP / (TP_Class3_rfTrainHP + FP_Class3_rfTrainHP);
recall_Class3_rfTrainHP = TP_Class3_rfTrainHP / (TP_Class3_rfTrainHP + FN_Class3_rfTrainHP);
f1Score_Class3_rfTrainHP = 2 * (precision_Class3_rfTrainHP * recall_Class3_rfTrainHP) / (precision_Class3_rfTrainHP + recall_Class3_rfTrainHP);
accuracy_Class3_rfTrainHP = (TP_Class3_rfTrainHP + TN_Class3_rfTrainHP) / sum(results_rfTrainHP(:));

disp(['Class 3 (Rain)']);
disp(['True Positive: ', num2str(TP_Class3_rfTrainHP)]);
disp(['False Negative: ', num2str(FN_Class3_rfTrainHP)]);
disp(['False Positive: ', num2str(FP_Class3_rfTrainHP)]);
disp(['True Negative: ', num2str(TN_Class3_rfTrainHP)]);
disp(['Precision: ', num2str(precision_Class3_rfTrainHP)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class3_rfTrainHP)]);
disp(['F1 Score: ', num2str(f1Score_Class3_rfTrainHP)]);
disp(['Accuracy: ', num2str(accuracy_Class3_rfTrainHP)]);
disp('----------------------');
%----------------------------------------------------
%Snow
TP_Class4_rfTrainHP = results_rfTrainHP(4, 4);

FN_Class4_rfTrainHP = sum(results_rfTrainHP(:, 4)) - TP_Class4_rfTrainHP;
FP_Class4_rfTrainHP = sum(results_rfTrainHP(4, :)) - TP_Class4_rfTrainHP;
TN_Class4_rfTrainHP = sum(results_rfTrainHP(:)) - (TP_Class4_rfTrainHP + FP_Class4_rfTrainHP + FN_Class4_rfTrainHP);
precision_Class4_rfTrainHP = TP_Class4_rfTrainHP / (TP_Class4_rfTrainHP + FP_Class4_rfTrainHP);
recall_Class4_rfTrainHP = TP_Class4_rfTrainHP / (TP_Class4_rfTrainHP + FN_Class4_rfTrainHP);
f1Score_Class4_rfTrainHP = 2 * (precision_Class4_rfTrainHP * recall_Class4_rfTrainHP) / (precision_Class4_rfTrainHP + recall_Class4_rfTrainHP);
accuracy_Class4_rfTrainHP = (TP_Class4_rfTrainHP + TN_Class4_rfTrainHP) / sum(results_rfTrainHP(:));

disp(['Class 4 (Snow)']);
disp(['True Positive: ', num2str(TP_Class4_rfTrainHP)]);
disp(['False Negative: ', num2str(FN_Class4_rfTrainHP)]);
disp(['False Positive: ', num2str(FP_Class4_rfTrainHP)]);
disp(['True Negative: ', num2str(TN_Class4_rfTrainHP)]);
disp(['Precision: ', num2str(precision_Class4_rfTrainHP)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class4_rfTrainHP)]);
disp(['F1 Score: ', num2str(f1Score_Class4_rfTrainHP)]);
disp(['Accuracy: ', num2str(accuracy_Class4_rfTrainHP)]);
disp('----------------------');

%----------------------------------------------------

%Sun
TP_Class5_rfTrainHP = results_rfTrainHP(5, 5);
FN_Class5_rfTrainHP = sum(results_rfTrainHP(:, 5)) - TP_Class5_rfTrainHP;
FP_Class5_rfTrainHP = sum(results_rfTrainHP(5, :)) - TP_Class5_rfTrainHP;
TN_Class5_rfTrainHP = sum(results_rfTrainHP(:)) - (TP_Class5_rfTrainHP + FP_Class5_rfTrainHP + FN_Class5_rfTrainHP);
precision_Class5_rfTrainHP = TP_Class5_rfTrainHP / (TP_Class5_rfTrainHP + FP_Class5_rfTrainHP);
recall_Class5_rfTrainHP = TP_Class5_rfTrainHP / (TP_Class5_rfTrainHP + FN_Class5_rfTrainHP);
f1Score_Class5_rfTrainHP = 2 * (precision_Class5_rfTrainHP * recall_Class5_rfTrainHP) / (precision_Class5_rfTrainHP + recall_Class5_rfTrainHP);
accuracy_Class5_rfTrainHP = (TP_Class5_rfTrainHP + TN_Class5_rfTrainHP) / sum(results_rfTrainHP(:));
disp(['Class 5 (Sun)']);
disp(['True Positive: ', num2str(TP_Class5_rfTrainHP)]);
disp(['False Negative: ', num2str(FN_Class5_rfTrainHP)]);
disp(['False Positive: ', num2str(FP_Class5_rfTrainHP)]);
disp(['True Negative: ', num2str(TN_Class5_rfTrainHP)]);
disp(['Precision: ', num2str(precision_Class5_rfTrainHP)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class5_rfTrainHP)]);
disp(['F1 Score: ', num2str(f1Score_Class5_rfTrainHP)]);
disp(['Accuracy: ', num2str(accuracy_Class5_rfTrainHP)]);
disp('----------------------');

%----------------------------------------------------
