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

p <- corPlot(data_numeric, cex = 1.2)
p


# # Remove the "date" variable from the data dataframe
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

# RH_7 removed
data_numeric <- dplyr::select(data_numeric, -RH_7)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

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

p <- corPlot(data_numeric, cex = 1.2)
p  



# RH_2 removed
data_numeric <- dplyr::select(data_numeric, -RH_2)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

p <- corPlot(data_numeric, cex = 1.2)
p  

# minute removed
data_numeric <- dplyr::select(data_numeric, -minute)
model_all <- lm(Appliances ~ ., data=data_numeric)
vif(model_all)

p <- corPlot(data_numeric, cex = 1.2)
p


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

# Preprocess the data
data_train <- data[1:round(0.7*nrow(data)),]
data_test <- data[(round(0.7*nrow(data))+1):nrow(data),]
x_train <- as.matrix(data_train[,-c(1,2)])
y_train <- as.matrix(data_train[,2])
x_test <- as.matrix(data_test[,-c(1,2)])
y_test <- as.matrix(data_test[,2])

# # Build the model
# model <- keras_model_sequential() %>%
#   layer_dense(units = 50, activation = "relu", input_shape = ncol(x_train)) %>%
#   layer_dense(units = 50, activation = "relu") %>%
#   layer_dense(units = 1)
# Build the model with L2 regularization
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train),
              kernel_regularizer = regularizer_l2(0.01)) %>%
  # layer_dense(units = 8, activation = "relu", kernel_regularizer = regularizer_l2(0.01)) %>%
  layer_dense(units = 4, activation = "relu", kernel_regularizer = regularizer_l2(0.01)) %>%
  layer_dense(units = 2, activation = "relu", kernel_regularizer = regularizer_l2(0.01)) %>%
  layer_dense(units = 1, activation = "relu", kernel_regularizer = regularizer_l2(0.01)) 
  # layer_dense(units = 128, activation = "relu", kernel_regularizer = regularizer_l2(0.01)) %>%
  # layer_dense(units = 64, activation = "relu", kernel_regularizer = regularizer_l2(0.01)) %>%
  # layer_dense(units = 32, activation = "relu", kernel_regularizer = regularizer_l2(0.01)) %>%
  # layer_dense(units = 1, activation = "relu", kernel_regularizer = regularizer_l2(0.01)) 
  # layer_dense(units = 128, activation = "relu", kernel_regularizer = regularizer_l2(0.01)) %>%
  # layer_dense(units = 64, activation = "relu", kernel_regularizer = regularizer_l2(0.01)) %>%
  # layer_dense(units = 32, activation = "relu", kernel_regularizer = regularizer_l2(0.01)) %>%
  # layer_dense(units = 8, activation = "relu", kernel_regularizer = regularizer_l2(0.01)) %>%
  # layer_dense(units = 1)

# Train the model with early stopping
early_stopping <- callback_early_stopping(monitor = "val_loss", patience = 10)

# Define a function to calculate MAPE
mape <- function(y_true, y_pred) {
  return(mean(abs((y_true - y_pred) / y_true)) * 100)
}

# Compile the model with MAPE as a metric
model %>% compile(loss = "mean_squared_error", optimizer = "adam", 
                  metrics = c("mean_absolute_error", mape))

# Fit the model and evaluate its performance
history <- model %>% fit(x_train, y_train, epochs = 100, batch_size = 32, 
                         validation_data = list(x_test, y_test))

# Plot the MAPE values during training
plot(history)


