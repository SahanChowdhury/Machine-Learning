%Training the data
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

%%

%Training decision tree model
%Training model decision tree using fitctree
%tic toc takes the time 

%Reference: https://uk.mathworks.com/help/matlab/ref/tic.html
tic
%Training the decision tree using fictree on the training data
dtTrain = fitctree(XTrain, YTrain);
toc


%%

%Predictions on the 20% testing set
predictions_dtTrain = predict(dtTrain, XTest);
%displaying first couple rows of predictions
head(predictions_dtTrain);
%Saving the training model predictions in a csv
%writematrix(predictions_dtTrain, 'predictions_dtTrain.csv');
%%
%Reference: https://uk.mathworks.com/help/matlab/ref/eq.html
%Reference: https://uk.mathworks.com/help/matlab/ref/sum.html

%Accuracy
%Summing all correct predictions by comparing to YTest (True values)
correctPredictions_dtTrain = sum(YTest == predictions_dtTrain);
%Total number 
totalPredictions_dtTrain = length(YTest);

% Calculate test accuracy
%By diving number of corect predictions by 
%Number of correct predictions/(lenght of test set = 292)
testAccuracy_dtTrain = correctPredictions_dtTrain /292;
AccuracyPercentage_dtTrain = testAccuracy_dtTrain*100
%Reference: https://uk.mathworks.com/help/matlab/ref/num2str.html
disp(['Test Accuracy: ' num2str(testAccuracy_dtTrain)]);


%Calculating the error (Amount of incorrect predictions)
error_dtTrain = 100- AccuracyPercentage_dtTrain;
error_dtTrain


%%
%Results
%Reference: https://uk.mathworks.com/help/stats/confusionmat.html
%calculating the confusion matrix given the true labels and predicted
%Where dt stands for Decision Tree
results_dtTrain = confusionmat(YTest, predictions_dtTrain);

%Displaying
results_dtTrain
%Total number of predictions made by model which should equate to the..
%lenght of Test set 
results_sum_dtTrain = sum(sum(results_dtTrain));
results_sum_dtTrain

%Heatmap visualization of the confusion matrix
figure;
dtTrain_Heatmap= heatmap(results_dtTrain);
%%
unique_labels = unique(data_copy.weather_labels);
unique_labels;
%The unique labels correspond as follows:
%1 -> Drizzle
%2 -> Fog
%3 -> Rain
%4 -> snow
%5 -> Sun

%%
% Performance metrics 
%dt stands for Decision Tree
% Class 1 (Drizzle)
% True Positive
TP_Class1_dtTrain = results_dtTrain(1, 1);

% False Negative
FN_Class1_dtTrain = sum(results_dtTrain(:, 1)) - TP_Class1_dtTrain;

% False Positive
FP_Class1_dtTrain = sum(results_dtTrain(1, :)) - TP_Class1_dtTrain;

% True Negative 
TN_Class1_dtTrain = sum(results_dtTrain(:)) - (TP_Class1_dtTrain + FP_Class1_dtTrain + FN_Class1_dtTrain);

% Precision
precision_Class1_dtTrain = TP_Class1_dtTrain / (TP_Class1_dtTrain + FP_Class1_dtTrain);

% Recall (Sensitivity)
recall_Class1_dtTrain = TP_Class1_dtTrain / (TP_Class1_dtTrain + FN_Class1_dtTrain);

% F1 Score
f1Score_Class1_dtTrain = 2 * (precision_Class1_dtTrain * recall_Class1_dtTrain) / (precision_Class1_dtTrain + recall_Class1_dtTrain);

% Accuracy
accuracy_Class1_dtTrain = (TP_Class1_dtTrain + TN_Class1_dtTrain) / sum(results_dtTrain(:));

% Displaying results for Class 1 (drizzle)
disp(['Class 1 (Drizzle)']);
disp(['True Positive: ', num2str(TP_Class1_dtTrain)]);
disp(['False Negative: ', num2str(FN_Class1_dtTrain)]);
disp(['False Positive: ', num2str(FP_Class1_dtTrain)]);
disp(['True Negative: ', num2str(TN_Class1_dtTrain)]);
disp(['Precision: ', num2str(precision_Class1_dtTrain)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class1_dtTrain)]);
disp(['F1 Score: ', num2str(f1Score_Class1_dtTrain)]);
disp(['Accuracy: ', num2str(accuracy_Class1_dtTrain)]);
disp('----------------------');


%---------------------------------------------
%Class 2 (Fog)

TP_Class2_dtTrain = results_dtTrain(2, 2);

FN_Class2_dtTrain = sum(results_dtTrain(:, 2)) - TP_Class2_dtTrain;

FP_Class2_dtTrain = sum(results_dtTrain(2, :)) - TP_Class2_dtTrain;

TN_Class2_dtTrain = sum(results_dtTrain(:)) - (TP_Class2_dtTrain + FP_Class2_dtTrain + FN_Class2_dtTrain);

precision_Class2_dtTrain = TP_Class2_dtTrain / (TP_Class2_dtTrain + FP_Class2_dtTrain);

recall_Class2_dtTrain = TP_Class2_dtTrain / (TP_Class2_dtTrain + FN_Class2_dtTrain);

f1Score_Class2_dtTrain = 2 * (precision_Class2_dtTrain * recall_Class2_dtTrain) / (precision_Class2_dtTrain + recall_Class2_dtTrain);

accuracy_Class2_dtTrain = (TP_Class2_dtTrain + TN_Class2_dtTrain) / sum(results_dtTrain(:));

% Displaying results for Class 2 (fog)
disp(['Class 2 (Fog)']);
disp(['True Positive: ', num2str(TP_Class2_dtTrain)]);
disp(['False Negative: ', num2str(FN_Class2_dtTrain)]);
disp(['False Positive: ', num2str(FP_Class2_dtTrain)]);
disp(['True Negative: ', num2str(TN_Class2_dtTrain)]);
disp(['Precision: ', num2str(precision_Class2_dtTrain)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class2_dtTrain)]);
disp(['F1 Score: ', num2str(f1Score_Class2_dtTrain)]);
disp(['Accuracy: ', num2str(accuracy_Class2_dtTrain)]);
disp('----------------------');

%---------------------------------------------
%Class 3 (rain)

TP_Class3_dtTrain = results_dtTrain(3, 3);

FN_Class3_dtTrain = sum(results_dtTrain(:, 3)) - TP_Class3_dtTrain;

FP_Class3_dtTrain = sum(results_dtTrain(3, :)) - TP_Class3_dtTrain;

TN_Class3_dtTrain = sum(results_dtTrain(:)) - (TP_Class3_dtTrain + FP_Class3_dtTrain + FN_Class3_dtTrain);

precision_Class3_dtTrain = TP_Class3_dtTrain / (TP_Class3_dtTrain + FP_Class3_dtTrain);

recall_Class3_dtTrain = TP_Class3_dtTrain / (TP_Class3_dtTrain + FN_Class3_dtTrain);

f1Score_Class3_dtTrain = 2 * (precision_Class3_dtTrain * recall_Class3_dtTrain) / (precision_Class3_dtTrain + recall_Class3_dtTrain);

accuracy_Class3_dtTrain = (TP_Class3_dtTrain + TN_Class3_dtTrain) / sum(results_dtTrain(:));

% Displaying results for Class 3 (rain)
disp(['Class 3 (Rain)']);
disp(['True Positive: ', num2str(TP_Class3_dtTrain)]);
disp(['False Negative: ', num2str(FN_Class3_dtTrain)]);
disp(['False Positive: ', num2str(FP_Class3_dtTrain)]);
disp(['True Negative: ', num2str(TN_Class3_dtTrain)]);
disp(['Precision: ', num2str(precision_Class3_dtTrain)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class3_dtTrain)]);
disp(['F1 Score: ', num2str(f1Score_Class3_dtTrain)]);
disp(['Accuracy: ', num2str(accuracy_Class3_dtTrain)]);
disp('----------------------');

%---------------------------------------------
%Class 4 (snow)
TP_Class4_dtTrain = results_dtTrain(4, 4);

FN_Class4_dtTrain = sum(results_dtTrain(:, 4)) - TP_Class4_dtTrain;

FP_Class4_dtTrain = sum(results_dtTrain(4, :)) - TP_Class4_dtTrain;

TN_Class4_dtTrain = sum(results_dtTrain(:)) - (TP_Class4_dtTrain + FP_Class4_dtTrain + FN_Class4_dtTrain);

precision_Class4_dtTrain = TP_Class4_dtTrain / (TP_Class4_dtTrain + FP_Class4_dtTrain);

recall_Class4_dtTrain = TP_Class4_dtTrain / (TP_Class4_dtTrain + FN_Class4_dtTrain);

f1Score_Class4_dtTrain = 2 * (precision_Class4_dtTrain * recall_Class4_dtTrain) / (precision_Class4_dtTrain + recall_Class4_dtTrain);

accuracy_Class4_dtTrain = (TP_Class4_dtTrain + TN_Class4_dtTrain) / sum(results_dtTrain(:));

% Displaying results for Class 4 (snow)
disp(['Class 4 (Snow)']);
disp(['True Positive: ', num2str(TP_Class4_dtTrain)]);
disp(['False Negative: ', num2str(FN_Class4_dtTrain)]);
disp(['False Positive: ', num2str(FP_Class4_dtTrain)]);
disp(['True Negative: ', num2str(TN_Class4_dtTrain)]);
disp(['Precision: ', num2str(precision_Class4_dtTrain)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class4_dtTrain)]);
disp(['F1 Score: ', num2str(f1Score_Class4_dtTrain)]);
disp(['Accuracy: ', num2str(accuracy_Class4_dtTrain)]);
disp('----------------------');



%---------------------------------------------
%Class 5 (Sun)

TP_Class5_dtTrain = results_dtTrain(5, 5);

FN_Class5_dtTrain = sum(results_dtTrain(:, 5)) - TP_Class5_dtTrain;

FP_Class5_dtTrain = sum(results_dtTrain(5, :)) - TP_Class5_dtTrain;

TN_Class5_dtTrain = sum(results_dtTrain(:)) - (TP_Class5_dtTrain + FP_Class5_dtTrain + FN_Class5_dtTrain);

precision_Class5_dtTrain = TP_Class5_dtTrain / (TP_Class5_dtTrain + FP_Class5_dtTrain);

recall_Class5_dtTrain = TP_Class5_dtTrain / (TP_Class5_dtTrain + FN_Class5_dtTrain);

f1Score_Class5_dtTrain = 2 * (precision_Class5_dtTrain * recall_Class5_dtTrain) / (precision_Class5_dtTrain + recall_Class5_dtTrain);

accuracy_Class5_dtTrain = (TP_Class5_dtTrain + TN_Class5_dtTrain) / sum(results_dtTrain(:));

% Displaying results for Class 5 (sun)
disp(['Class 5 (Sun)']);
disp(['True Positive: ', num2str(TP_Class5_dtTrain)]);
disp(['False Negative: ', num2str(FN_Class5_dtTrain)]);
disp(['False Positive: ', num2str(FP_Class5_dtTrain)]);
disp(['True Negative: ', num2str(TN_Class5_dtTrain)]);
disp(['Precision: ', num2str(precision_Class5_dtTrain)]);
disp(['Recall (Sensitivity): ', num2str(recall_Class5_dtTrain)]);
disp(['F1 Score: ', num2str(f1Score_Class5_dtTrain)]);
disp(['Accuracy: ', num2str(accuracy_Class5_dtTrain)]);
disp('----------------------');


%%
tic
rng(1)
%Decision Tree Hyperparameter optimization

decisionTree_HP = fitctree(XTrain, YTrain, 'OptimizeHyperparameters','auto');
toc
%%
%saving and loading
% Save the decision tree model
save('decisionTree_HP.mat', 'decisionTree_HP');


%end of traing random forest model