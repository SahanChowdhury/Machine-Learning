%Training Model 2 Random Forest
rng(1)
data_copy = readtable('data_copy.csv')

%Reference: https://uk.mathworks.com/matlabcentral/answers/377839-split-training-data-and-testing-data% 
% Reference: https://www.mathworks.com/help/stats/cvpartition.html
%Reference: https://uk.mathworks.com/help/matlab/ref/rng.html
%Using random number generator so data results is reproducible

%creating a crossvalidation partition using 'holdout' method
cv = cvpartition(size(data_copy,1), 'HoldOut', 0.2)
%Assinging the index of the test set to to variable name idx
idx = cv.test;

% Splitting the data into training and testing sets using the partition
% ~idx takes the negative 
trainingData = data_copy(~idx,:);
testingData = data_copy(idx,:);
%Saving Testing data into a matlab file for later use when predicting
save('test_data.mat', 'testingData')
save('training_data.mat', 'trainingData');
% Defining feature columns and target column
X = {'Year', 'Month', 'Day', 'precipitation', 'temp_max', 'temp_min', 'wind', 'temp_range', 'Winter', 'Summer', 'Spring', 'Autumn'};
Y = 'weather_labels';
% Separate features (X) and labels (Y) in the training set
XTrain = trainingData(:, X);
YTrain = trainingData.(Y);

% Separate features (X) and labels (Y) in the testing set
XTest = testingData(:, X);
YTest = testingData.(Y);

% Display the sizes of the training and testing sets
disp('Number of samples in the training set: ');
disp(size(trainingData));
disp('Number of samples in the testing set: ');
disp(size(testingData));



%Model 2: Random Forest

%Training Random forest model
%Training Random forest using fitenemble
tic
%Random Number generator
%Reference: https://uk.mathworks.com/help/matlab/ref/rng.html

%Reference:https://uk.mathworks.com/help/stats/select-predictors-for-random-forests.html
%Where rf stands for Random Forest
rfTrain = fitensemble(XTrain, YTrain, 'Bag', 100, 'Tree', 'Type', 'classification');
toc




%Predictions on the 20% testing set
predictions_rfTrain = predict(rfTrain, XTest);
%displaying first couple rows of predictions
head(predictions_rfTrain);
%Saving the training model predictions in a csv
%writematrix(predictions_dtTrain, 'predictions_dtTrain.csv');


%Accuracy
%Summing all correct predictions by comparing to YTest (True values)
correctPredictions_rfTrain = sum(YTest == predictions_rfTrain);
%Total number 
totalPredictions_rfTrain = length(YTest);

%By diving number of corect predictions by 
%Number of correct predictions/(lenght of test set = 292)
testAccuracy_rfTrain = correctPredictions_rfTrain /292;
AccuracyPercentage_rfTrain = testAccuracy_rfTrain*100
disp(['Test Accuracy: ' num2str(testAccuracy_rfTrain)]);


%Results

results_rfTrain = confusionmat(YTest, predictions_rfTrain);
results_rfTrain
results_sum_rfTrain = sum(sum(results_rfTrain));
results_sum_rfTrain

figure;
rfTrain_Heatmap= heatmap(results_rfTrain);


%Evaluation metrics

%Rain
TP_Class1_rfTrain = results_rfTrain(1, 1);
FN_Class1_rfTrain = sum(results_rfTrain(:, 1)) - TP_Class1_rfTrain;
FP_Class1_rfTrain = sum(results_rfTrain(1, :)) - TP_Class1_rfTrain;
TN_Class1_rfTrain = sum(results_rfTrain(:)) - (TP_Class1_rfTrain + FP_Class1_rfTrain + FN_Class1_rfTrain);
precision_Class1_rfTrain = TP_Class1_rfTrain / (TP_Class1_rfTrain + FP_Class1_rfTrain);
recall_Class1_rfTrain = TP_Class1_rfTrain / (TP_Class1_rfTrain + FN_Class1_rfTrain);
f1Score_Class1_rfTrain = 2 * (precision_Class1_rfTrain * recall_Class1_rfTrain) / (precision_Class1_rfTrain + recall_Class1_rfTrain);
accuracy_Class1_rfTrain = (TP_Class1_rfTrain + TN_Class1_rfTrain) / sum(results_rfTrain(:));

disp(['Class 1 (Drizzle)']);
disp(['True Positive: ', num2str(TP_Class1_rfTrain)]);
disp(['False Negative: ', num2str(FN_Class1_rfTrain)]);
disp(['False Positive: ', num2str(FP_Class1_rfTrain)]);
disp(['True Negative: ', num2str(TN_Class1_rfTrain)]);
disp(['Precision: ', num2str(precision_Class1_rfTrain)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class1_rfTrain)]);
disp(['F1 Score: ', num2str(f1Score_Class1_rfTrain)]);
disp(['Accuracy: ', num2str(accuracy_Class1_rfTrain)]);
disp('----------------------');
%----------------------------------------------------
%Fog
TP_Class2_rfTrain = results_rfTrain(2, 2);
FN_Class2_rfTrain = sum(results_rfTrain(:, 2)) - TP_Class2_rfTrain;
FP_Class2_rfTrain = sum(results_rfTrain(2, :)) - TP_Class2_rfTrain;
TN_Class2_rfTrain = sum(results_rfTrain(:)) - (TP_Class2_rfTrain + FP_Class2_rfTrain + FN_Class2_rfTrain);
precision_Class2_rfTrain = TP_Class2_rfTrain / (TP_Class2_rfTrain + FP_Class2_rfTrain);
recall_Class2_rfTrain = TP_Class2_rfTrain / (TP_Class2_rfTrain + FN_Class2_rfTrain);
f1Score_Class2_rfTrain = 2 * (precision_Class2_rfTrain * recall_Class2_rfTrain) / (precision_Class2_rfTrain + recall_Class2_rfTrain);
accuracy_Class2_rfTrain = (TP_Class2_rfTrain + TN_Class2_rfTrain) / sum(results_rfTrain(:));

disp(['Class 2 (Fog)']);
disp(['True Positive: ', num2str(TP_Class2_rfTrain)]);
disp(['False Negative: ', num2str(FN_Class2_rfTrain)]);
disp(['False Positive: ', num2str(FP_Class2_rfTrain)]);
disp(['True Negative: ', num2str(TN_Class2_rfTrain)]);
disp(['Precision: ', num2str(precision_Class2_rfTrain)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class2_rfTrain)]);
disp(['F1 Score: ', num2str(f1Score_Class2_rfTrain)]);
disp(['Accuracy: ', num2str(accuracy_Class2_rfTrain)]);
disp('----------------------');


%----------------------------------------------------
%Rain
TP_Class3_rfTrain = results_rfTrain(3, 3);
FN_Class3_rfTrain = sum(results_rfTrain(:, 3)) - TP_Class3_rfTrain;
FP_Class3_rfTrain = sum(results_rfTrain(3, :)) - TP_Class3_rfTrain;
TN_Class3_rfTrain = sum(results_rfTrain(:)) - (TP_Class3_rfTrain + FP_Class3_rfTrain + FN_Class3_rfTrain);
precision_Class3_rfTrain = TP_Class3_rfTrain / (TP_Class3_rfTrain + FP_Class3_rfTrain);
recall_Class3_rfTrain = TP_Class3_rfTrain / (TP_Class3_rfTrain + FN_Class3_rfTrain);
f1Score_Class3_rfTrain = 2 * (precision_Class3_rfTrain * recall_Class3_rfTrain) / (precision_Class3_rfTrain + recall_Class3_rfTrain);
accuracy_Class3_rfTrain = (TP_Class3_rfTrain + TN_Class3_rfTrain) / sum(results_rfTrain(:));
disp(['Class 3 (Raim)']);
disp(['True Positive: ', num2str(TP_Class3_rfTrain)]);
disp(['False Negative: ', num2str(FN_Class3_rfTrain)]);
disp(['False Positive: ', num2str(FP_Class3_rfTrain)]);
disp(['True Negative: ', num2str(TN_Class3_rfTrain)]);
disp(['Precision: ', num2str(precision_Class3_rfTrain)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class3_rfTrain)]);
disp(['F1 Score: ', num2str(f1Score_Class3_rfTrain)]);
disp(['Accuracy: ', num2str(accuracy_Class3_rfTrain)]);
disp('----------------------');



%----------------------------------------------------

%Snow
TP_Class4_rfTrain = results_rfTrain(4, 4);
FN_Class4_rfTrain = sum(results_rfTrain(:, 4)) - TP_Class4_rfTrain;
FP_Class4_rfTrain = sum(results_rfTrain(4, :)) - TP_Class4_rfTrain;
TN_Class4_rfTrain = sum(results_rfTrain(:)) - (TP_Class4_rfTrain + FP_Class4_rfTrain + FN_Class4_rfTrain);
precision_Class4_rfTrain = TP_Class4_rfTrain / (TP_Class4_rfTrain + FP_Class4_rfTrain);
recall_Class4_rfTrain = TP_Class4_rfTrain / (TP_Class4_rfTrain + FN_Class4_rfTrain);
f1Score_Class4_rfTrain = 2 * (precision_Class4_rfTrain * recall_Class4_rfTrain) / (precision_Class4_rfTrain + recall_Class4_rfTrain);
accuracy_Class4_rfTrain = (TP_Class4_rfTrain + TN_Class4_rfTrain) / sum(results_rfTrain(:));
disp(['Class 4 (Snow)']);
disp(['True Positive: ', num2str(TP_Class4_rfTrain)]);
disp(['False Negative: ', num2str(FN_Class4_rfTrain)]);
disp(['False Positive: ', num2str(FP_Class4_rfTrain)]);
disp(['True Negative: ', num2str(TN_Class4_rfTrain)]);
disp(['Precision: ', num2str(precision_Class4_rfTrain)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class4_rfTrain)]);
disp(['F1 Score: ', num2str(f1Score_Class4_rfTrain)]);
disp(['Accuracy: ', num2str(accuracy_Class4_rfTrain)]);
disp('----------------------');




%----------------------------------------------------

%Sun
TP_Class5_rfTrain = results_rfTrain(5, 5);
FN_Class5_rfTrain = sum(results_rfTrain(:, 5)) - TP_Class5_rfTrain;
FP_Class5_rfTrain = sum(results_rfTrain(5, :)) - TP_Class5_rfTrain;
TN_Class5_rfTrain = sum(results_rfTrain(:)) - (TP_Class5_rfTrain + FP_Class5_rfTrain + FN_Class5_rfTrain);
precision_Class5_rfTrain = TP_Class5_rfTrain / (TP_Class5_rfTrain + FP_Class5_rfTrain);
recall_Class5_rfTrain = TP_Class5_rfTrain / (TP_Class5_rfTrain + FN_Class5_rfTrain);
f1Score_Class5_rfTrain = 2 * (precision_Class5_rfTrain * recall_Class5_rfTrain) / (precision_Class5_rfTrain + recall_Class5_rfTrain);
accuracy_Class5_rfTrain = (TP_Class5_rfTrain + TN_Class5_rfTrain) / sum(results_rfTrain(:));

disp(['Class 5 (Sun)']);
disp(['True Positive: ', num2str(TP_Class5_rfTrain)]);
disp(['False Negative: ', num2str(FN_Class5_rfTrain)]);
disp(['False Positive: ', num2str(FP_Class5_rfTrain)]);
disp(['True Negative: ', num2str(TN_Class5_rfTrain)]);
disp(['Precision: ', num2str(precision_Class5_rfTrain)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class5_rfTrain)]);
disp(['F1 Score: ', num2str(f1Score_Class5_rfTrain)]);
disp(['Accuracy: ', num2str(accuracy_Class5_rfTrain)]);
disp('----------------------');



%----------------------------------------------------
disp(['Accuracy of normal model']);
AccuracyPercentage_rfTrain
%%
rng(1)
tic

RandomForest_HP = fitcensemble(XTrain, YTrain, 'OptimizeHyperparameters', 'auto', 'Method', 'bag');

toc

%saving and loading
% Save the decision tree model
%Reference: 

save('RandomForest_HP.mat', 'RandomForest_HP');
%%

