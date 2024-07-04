

%% -- Data Importing
clc, clear all, close all, warning off

% Loading the data from local file
% reference - https://uk.mathworks.com/help/matlab/ref/readtable.html
data = readtable('seattle-weather.csv');

% Displaying the first few rows of the dataset
disp(['First couple rows of the data:']);
head(data);

% Displaying the last few rows of the dataset
disp(['Last couple rows of the data:']);
tail(data);

% Displaying the column names
disp(['Data column names']);
disp(data.Properties.VariableNames);

%Displaying number of rows and columns
disp(['Number of Rows and Columns:']);
disp(size(data));

%Displaying the timeframe of data
disp(['The dataset is from:']);
disp(min(data.date));
disp(['Till:']);
disp(max(data.date));

%This Dataset is a weather dataset of Seattle, USA. which contains.....
%'precipitation','temp_max','temp_min','wind','weather' values and...
%types for Seattle from 01/01/2012 - 31/12/2015 everyday
%%

%% -- Data Processing 

%General Data Statistics

%Displaying summary statistics of dataset
disp('Summary Statistics:' )
summary(data);
%4 numeric weather type columns
%1 timeframe column (date)
%1 categorical column (weather)

%Reference: https://uk.mathworks.com/help/matlab/ref/std.html
%Computing Standard deviations

disp('Standard dev of percipitation column:' )
disp(std(data.precipitation));
disp('Standard dev of temp_max column:' )
disp(std(data.temp_max));
disp('Standard dev of temp_min column:' )
disp(std(data.temp_min));
disp('Standard dev of wind speed column:' )
disp(std(data.wind));

%These were the stds
%Percipitation: 6.6802
%temp_max: 7.3498
%temp_min:5.0230
%wind speed 1.4378

%From this we can observe wind column has the lowest std = low variability
%temp_max has the highest
%standardization should reduce this variability 

%Checking for duplicates through the date column 
%Reference: https://uk.mathworks.com/matlabcentral/answers/688889-how-to-convert-a-column-in-a-table-to-date-format-for-plotting-a-time-series
data.date = datetime(data.date, 'InputFormat', 'yyyy-MM-dd');
%Reference: https://uk.mathworks.com/matlabcentral/answers/19042-finding-duplicate-values-per-column
uniqueDate = unique(data.date);
[countOfDate] = histcounts(data.date, uniqueDate);
indexToRepeatedValue = (countOfDate ~= 1);
repeatedValues = uniqueDate(indexToRepeatedValue);
numberOfAppearancesOfRepeatedValues = countOfDate(indexToRepeatedValue);
%Theres no duplicates, and the dataset is a continuous time series

% Create a copy of the original data
%This is where we will alter the data
data_copy = readtable('seattle-weather.csv');


%Reference: https://blogs.mathworks.com/student-lounge/2023/01/11/weather-forecasting-in-matlab-for-wids-datathon-2023/
%Extract Day, month, year to make compatible for ml algorithms
data_copy.Day = data_copy.date.Day;
data_copy.Month = data_copy.date.Month;
data_copy.Year = data_copy.date.Year;
%Removing date column
data_copy.date = [];

%Reference: https://uk.mathworks.com/help/matlab/ref/table.movevars.html
%Moving weather column to the end
data_copy = movevars(data_copy, "weather", "After", "Year");
head(data_copy)

% Checking for missing values
disp(['Missing Values Count for Each Variable:']);
disp(sum(ismissing(data)));
%Since there is no missing values we dont need to replace/remove anything

%Displaying all the different weather types in the dataset
unique_weather = unique(data_copy.weather);
disp('These are the different weathers')
disp(unique_weather);
%The unique weather types are 
%drizzle
%fog
%snow
%rain
%sun

%%

% Asigning all unique weather types to a number 
%since there are 5 weather types i'm assinging all from 1-5

% Reference: https://www.mathworks.com/help/matlab/ref/containers.map.html
weather_mapping = containers.Map(unique_weather, [1, 2, 3, 4, 5]);

% Assign numeric labels to the 'weather_labels' column
% Reference: https://www.mathworks.com/help/matlab/ref/cell2mat.html
data_copy.weather_labels = cell2mat(values(weather_mapping, data_copy.weather));
%removing weather column as we dont want categorical data
data_copy.weather= []


%Creating a column called temp_range which calculates temp range everyday
data_copy.temp_range = data_copy.temp_max - data_copy.temp_min;

% Display the updated data_copy table
disp(['Updated data_copy with temp_range and weather)labels:']);



% Check for zero values in numeric columns of data_copy
% Reference : https://uk.mathworks.com/matlabcentral/answers/838378-how-to-delect-the-zero-values-in-table
disp("Columns with Zero Values in data:");
% Display columns with zero values in data_copy
disp(sum(data_copy{:, {'precipitation', 'temp_max', 'temp_min', 'wind','Day','Month','Year','weather_labels','temp_range'}} == 0));

%there is 838 zero values in percipitation column
%2 temp_max column
%16 temp_min column
%zero values dont need to be removed as its a weather dataset


% Create binary columns for each season to improve model 
%Reference: https://uk.mathworks.com/help/matlab/ref/double.ismember.html

%Winter is assigned to months 1,2,3,12
data_copy.Winter = double(ismember(data_copy.Month, [1, 2, 3, 12]));
%Summer is assigned to months 6,7,8
data_copy.Summer = double(ismember(data_copy.Month, [6, 7, 8]));
%Autumn is assigned to months 9,10,11
data_copy.Autumn = double(ismember(data_copy.Month, [9, 10, 11]));
%Spring is assigned to months 4,5
data_copy.Spring = double(ismember(data_copy.Month, [4, 5]));
%information obtained : https://www.timeanddate.com/calendar/seasons.html?n=234


%reference: https://uk.mathworks.com/help/matlab/ref/table.movevars.html
%moving the target column to the end of the table
data_copy = movevars(data_copy, "weather_labels", "After", "temp_range");

% Display the updated data_copy table
disp(['Updated data_copy with standarised columsns, temp_range column,weather)labels, and additional binary season columns:']);
head(data_copy)
%%
%%Standardizing the weather predictor columns 
% Reference: https://github.com/vighnesh32/Machine-Learning-Project/blob/main/diabholdout.m
% Standardization of precipitation column
mean_precipitation = mean(data_copy.precipitation);
std_precipitation = std(data_copy.precipitation);
stan_precipitation = (data_copy.precipitation - mean_precipitation) / std_precipitation;
data_copy.precipitation = stan_precipitation;

% Reference: https://github.com/vighnesh32/Machine-Learning-Project/blob/main/diabholdout.m
% Standardization of temp_max column
mean_temp_max = mean(data_copy.temp_max);
std_temp_max = std(data_copy.temp_max);
stan_temp_max = (data_copy.temp_max - mean_temp_max) / std_temp_max;
data_copy.temp_max = stan_temp_max;

% Reference: https://github.com/vighnesh32/Machine-Learning-Project/blob/main/diabholdout.m
% Standardization of temp_min column
mean_temp_min = mean(data_copy.temp_min);
std_temp_min = std(data_copy.temp_min);
stan_temp_min = (data_copy.temp_min - mean_temp_min) / std_temp_min;
data_copy.temp_min = stan_temp_min;

% Reference: https://github.com/vighnesh32/Machine-Learning-Project/blob/main/diabholdout.m
% Standardization of wind column
mean_wind = mean(data_copy.wind);
std_wind = std(data_copy.wind);
stan_wind = (data_copy.wind - mean_wind) / std_wind;
data_copy.wind = stan_wind;

% Displaying the first few rows of the updated dataset
head(data_copy);

%Save it as a csv
writetable(data_copy, 'data_copy.csv');

%% -- Data Visualization


%Reference: https://uk.mathworks.com/videos/how-to-make-subplots-in-matlab-using-tiledlayout-1599239984171.html
%Plotting scatter plots for all variables
tiledlayout('flow')

% Precipitation vs Temperature Min
nexttile
scatter(data_copy.precipitation, data_copy.temp_min)
xlabel('Precipitation')
ylabel('Temperature Min')
title('Precipitation vs Temperature Min')

%Precipitation vs Temperature Max
nexttile
scatter(data_copy.precipitation, data_copy.temp_max)
xlabel('Precipitation')
ylabel('Temperature Max')
title('Precipitation vs Temperature Max')

%Precipitation vs Wind Speed
nexttile
scatter(data_copy.precipitation, data_copy.wind)
xlabel('Precipitation')
ylabel('Wind Speed')
title('Precipitation vs Wind Speed')

%Temperature Max vs Temperature Min
nexttile
scatter(data_copy.temp_max, data_copy.temp_min)
xlabel('Temperature Max')
ylabel('Temperature Min')
title('Temperature Max vs Temperature Min')

%Temperature Max vs Wind Speed
nexttile
scatter(data_copy.temp_max, data_copy.wind)
xlabel('Temperature Max')
ylabel('Wind Speed')
title('Temperature Max vs Wind Speed')

%Wind Speed vs Temperature Min
nexttile
scatter(data_copy.wind, data_copy.temp_min)
xlabel('Wind Speed')
ylabel('Temperature Min')
title('Wind Speed vs Temperature Min')


%Boxplot for different weather types and temp min vs temp max
% Boxplot for temp_min for different weather types
figure;
boxplot(data.temp_min, data.weather, 'Labels', unique(data.weather));
xlabel('Weather Type');
ylabel('Temperature Min');
title('Boxplot: Temperature Min across Weather Types');

% Boxplot for temp_max for different weather types
figure;
boxplot(data.temp_max, data.weather, 'Labels', unique(data.weather));
xlabel('Weather Type');
ylabel('Temperature Max');
title('Boxplot: Temperature Max across Weather Types');


%Correlation Matrix between predictors
%Reference: https://uk.mathworks.com/help/matlab/ref/heatmap.html
numeric_columns_copy = data_copy{:, {'precipitation', 'temp_max', 'temp_min', 'wind'}};
figure(3)
corr = corr(numeric_columns_copy);
xvalues = {'precipitation', 'temp_max', 'temp_min', 'wind'};
yvalues = {'precipitation', 'temp_max', 'temp_min', 'wind'};
h = heatmap(xvalues, yvalues,corr);



%Subplots histograms for every variable count
%Reference: https://uk.mathworks.com/help/matlab/ref/matlab.graphics.chart.primitive.histogram.html
%Reference: https://uk.mathworks.com/help/matlab/ref/subplot.html
figure(4)

subplot(2, 2, 1);
histogram(data.precipitation);
xlabel('Precipitation');
ylabel('Counts');
title('Precipitation Count');

subplot(2, 2, 2);
histogram(data.temp_max);
xlabel('Temp Max');
ylabel('Counts');
title('Temp Max Count');

subplot(2, 2, 3);
histogram(data.temp_min);
xlabel('Temp Min');
ylabel('Counts');
title('Temp Min count ');

subplot(2, 2, 4);
histogram(data.wind);
xlabel('Wind');
ylabel('Counts');
title('Wind Count');
%%
%Temp_range over time
figure(5)
plot(data_copy.Year, data_copy.temp_range, 'o-', 'LineWidth', 2);
xlabel('Year');
ylabel('Temperature Range');
title('Temperature Range Over Years');

%%
%%
%Boxplots for each predictors

%Box plot for precipitation
figure(6)
subplot(2,3,1)
boxplot(data_copy.precipitation);
title('Box Plot for Precipitation');

%Box plot for temp_max
subplot(2,3,2)
boxplot(data_copy.temp_max);
title('Box Plot for Temperature Max');

%Box plot for temp_min
subplot(2,3,3)
boxplot(data_copy.temp_min);
title('Box Plot for Temperature Min');

%Box plot for wind

subplot(2,3,4)
boxplot(data_copy.wind);
title('Box Plot for Wind');

%Box plot for temp_range
subplot(2,3,5)
boxplot(data_copy.temp_range);
title('Box Plot for Temperature range');

%end of data preprocessing