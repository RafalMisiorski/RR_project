library(readr)
library(neuralnet)
library(xgboost)
library(caTools)
library(car)
library(quantmod)
library(MASS)
library(corrplot)
library(keras)
library(tensorflow)
library(psych)
library(lubridate)
library(dplyr)
Sys.setenv(LANG = "en")

#First load the data
data <- read_csv("energydata_complete.csv")

#EDA
# Check for missing values
print(sum(is.na(data)))

# Summarize the data
summary(data)

# check the data types of each column
sapply(data, class)

# Extract the year, month, day, hour, minute, and second from the date-time variable
data$year <- year(data$date)
data$month <- month(data$date)
data$day <- day(data$date)
data$hour <- hour(data$date)
data$minute <- minute(data$date)
data$second <- second(data$date)

# Create new variable indicating the day of the week
data$day_of_week <- wday(data$date, label = TRUE)

# Create new variable indicating the hour of the day
data$hour_of_day <- hour(data$date)

# Create new variable indicating the time of the day (morning, afternoon, evening)
data$time_of_day <- ifelse(data$hour_of_day >= 6 & data$hour_of_day < 12, "morning",
                           ifelse(data$hour_of_day >= 12 & data$hour_of_day < 18, "afternoon", "evening"))

# Filter out non-numeric columns
numeric_columns <- sapply(data, is.numeric)
data_numeric <- data[, numeric_columns]

model_all <- lm(Appliances ~ ., data=data_numeric) 

summary(model_all)

summary(data_numeric$rv2)

vif(model_all)

# p <- corPlot(data_numeric, cex = 1.2)
# p


# Remove the "date" variable from the data dataframe
data_numeric <- dplyr::select(data_numeric, -second, -year, -hour_of_day, -rv1, -rv2)

model_all <- lm(Appliances ~ ., data=data_numeric) 

summary(model_all)
vif(model_all)

#I will remove TDewpoint,
data_numeric <- dplyr::select(data_numeric, -Tdewpoint)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

#T9 removed
data_numeric <- dplyr::select(data_numeric, -T9)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

#Month removed
data_numeric <- dplyr::select(data_numeric, -month)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

#T6 removed
data_numeric <- dplyr::select(data_numeric, -T6)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

#T2 removed
data_numeric <- dplyr::select(data_numeric, -T2)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

#T7 removed
data_numeric <- dplyr::select(data_numeric, -T7)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

#RH_4 removed
data_numeric <- dplyr::select(data_numeric, -RH_4)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

# #RH_1 removed
data_numeric <- dplyr::select(data_numeric, -RH_1)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

# T1 removed
data_numeric <- dplyr::select(data_numeric, -T1)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

# RH_7 removed
data_numeric <- dplyr::select(data_numeric, -RH_7)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

# # RH_7 removed
# data_numeric <- dplyr::select(data_numeric, -RH_7)
# model_all <- lm(Appliances ~ ., data=data_numeric)
# vif(model_all)

# RH_3 removed
data_numeric <- dplyr::select(data_numeric, -RH_3)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

# RH_8 removed
data_numeric <- dplyr::select(data_numeric, -RH_8)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

# T5 removed
data_numeric <- dplyr::select(data_numeric, -T5)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

# RH_6 removed
data_numeric <- dplyr::select(data_numeric, -RH_6)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

# T3 removed
data_numeric <- dplyr::select(data_numeric, -T3)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

# T4 removed
data_numeric <- dplyr::select(data_numeric, -T4)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

corPlot(data_numeric, cex = 1.2, margin = c(3,3,3,3))


# # Remove the "date" variable from the data dataframe
data <- select(data, -date, -day_of_week)
data <- select(data, -day_of_week)
data <- select(data, -time_of_day)
# 

# Create a vector of predictors
predictors <- names(data_numeric)
predictors <- predictors[predictors != "Appliances"]

# Subset the data to include only the predictors
subset_data <- data_numeric[, predictors]

# Standardize the predictors
subset_data <- scale(subset_data)

# Re-combine the standardized predictors with the "Appliances" variable
data_numeric[, predictors] <- subset_data


data <- data_numeric
# split the data into training and testing data
train_indices <- sample(1:nrow(data), 0.8*nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]
# Define the predictor variables (all columns except 'Appliances' and 'date') and the response variable ('Appliances')
predictors <- names(data)[-c(1,2)]
response <- "Appliances"




#Then convert the data into DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(train_data[predictors]), label = unlist(train_data[response]))
dtest <- xgb.DMatrix(data = as.matrix(test_data[predictors]))

# Train the XGBoost model on the training data
xgb_model <- xgboost(data = dtrain,objective = "reg:squarederror",lambda = 0.8,alpha =0.9, nrounds = 100,colsample_bytree = 0.6, gamma=0.85 , eta = 0.036, max_depth = 25, min_child_weight = 1, subsample=0.8, num_parallel_tree = 4, early_stopping_rounds = 50)

# xgb_model <- xgboost(data = dtrain,objective = "reg:squarederror",lambda = 0.8,alpha =0.8, nrounds = 100,colsample_bytree = 0.5, gamma=0.8 , eta = 0.03125, max_depth = 25, min_child_weight = 1, subsample=0.8)

#    nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
# 58     100        25 0.04   0.8              0.5                1       0.8


# Make predictions on the test data
predictions <- predict(xgb_model, newdata = dtest)
# predictions

#Print the dimensions of the variables
print(length(predictions))
print(length(test_response))

# Extract the response variable from the test data as a vector
test_response <- unlist(test_data[response])

# Evaluate the model's performance
print(mean((predictions - test_response)^2,na.rm = TRUE))


# Calculate the absolute percentage error
ape <- abs((test_response - predictions) / test_response)

# Calculate the mean absolute percentage error
MAPE <- mean(ape) * 100

# Print the result
print(MAPE)
