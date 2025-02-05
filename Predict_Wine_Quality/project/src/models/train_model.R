rm(list = ls())
library(boot)
library(randomForest)
library(xgboost)
library(dplyr)
library(rpart)
library(gbm)
library(caret)

# Load the training datasets
trainRed <- read.csv("project/volume/data/raw/trainRed.csv")
trainWhite <- read.csv("project/volume/data/raw/trainWhite.csv")
testRed <- read.csv("project/volume/data/raw/testRed.csv")
testWhite <- read.csv("project/volume/data/raw/testWhite.csv")


# Remove NA values
trainRed <- na.omit(trainRed)
trainWhite <- na.omit(trainWhite)



# Split into training and validation sets
set.seed(123)
trainIndexRed <- sample(1:nrow(trainRed), 0.8*nrow(trainRed))
trainSetRed <- trainRed[trainIndexRed, ]
validSetRed <- trainRed[-trainIndexRed, ]

trainIndexWhite <- sample(1:nrow(trainWhite), 0.8*nrow(trainWhite))
trainSetWhite <- trainWhite[trainIndexWhite, ]
validSetWhite <- trainWhite[-trainIndexWhite, ]

# Train the RandomForest models
# Random Forest with different parameters
rfModelRed <- randomForest(quality ~ ., data=trainSetRed, ntree=100, mtry=3, importance=TRUE)
rfModelWhite <- randomForest(quality ~ ., data=trainSetWhite, ntree=100, mtry=3, importance=TRUE)

# Predict on the validation set
predictionsRfRed <- predict(rfModelRed, newdata=validSetRed)
predictionsRfWhite <- predict(rfModelWhite, newdata=validSetWhite)

# Compute RMSE for RandomForest
rmseRfRed <- sqrt(mean((predictionsRfRed - validSetRed$quality)^2))
rmseRfWhite <- sqrt(mean((predictionsRfWhite - validSetWhite$quality)^2))

# Train the XGBoost models
dataRedMatrix <- xgb.DMatrix(data=as.matrix(trainSetRed[, -which(names(trainSetRed) == "quality")]), label=trainSetRed$quality)
dataWhiteMatrix <- xgb.DMatrix(data=as.matrix(trainSetWhite[, -which(names(trainSetWhite) == "quality")]), label=trainSetWhite$quality)
params <- list(  objective           = "reg:squarederror",#objective function
                 gamma               =0.2,#minimum loss function to make a split
                 booster             = "gbtree",#type of boosting to use
                 eval_metric         = "rmse",#evaluation metric
                 eta                 = 0.05,#How much each tree should contribute to the final tree
                 max_depth           = as.integer(10),#maximum depth of the tree
                 min_child_weight    = 5,
                 subsample           = 0.9,#proportion of the observation to create the tree
                 colsample_bytree    = 0.8,#proportion of columns to create each tree
                 alpha               = 0.5
                 
)

# Set the number of cross-validation folds
nfold <- 5

# Set the number of boosting rounds
nrounds <- 100

# Cross-validation for Red Wine
cvRed <- xgb.cv(params = params, data = dataRedMatrix, nfold = nfold, nrounds = nrounds, early_stopping_rounds = 10, print_every_n = 10)
optimal_nrounds_Red <- cvRed$best_iteration

# Cross-validation for White Wine
cvWhite <- xgb.cv(params = params, data = dataWhiteMatrix, nfold = nfold, nrounds = nrounds, early_stopping_rounds = 10, print_every_n = 10)
optimal_nrounds_White <- cvWhite$best_iteration

# Print the optimal number of boosting rounds
cat("Optimal number of boosting rounds for Red Wine:", optimal_nrounds_Red, "\n")
cat("Optimal number of boosting rounds for White Wine:", optimal_nrounds_White, "\n")

xgbModelRed <- xgb.train(params=params, data=dataRedMatrix, nrounds=100)
xgbModelWhite <- xgb.train(params=params, data=dataWhiteMatrix, nrounds=100)

# Predict on the validation set using XGBoost
validRedMatrix <- xgb.DMatrix(data=as.matrix(validSetRed[, -which(names(validSetRed) == "quality")]), label=validSetRed$quality)
validWhiteMatrix <- xgb.DMatrix(data=as.matrix(validSetWhite[, -which(names(validSetWhite) == "quality")]), label=validSetWhite$quality)


predictionsXgbRed <- predict(xgbModelRed, newdata=validRedMatrix)
predictionsXgbWhite <- predict(xgbModelWhite, newdata=validWhiteMatrix)

# Compute RMSE for XGBoost
rmseXgbRed <- sqrt(mean((predictionsXgbRed - validSetRed$quality)^2))
rmseXgbWhite <- sqrt(mean((predictionsXgbWhite - validSetWhite$quality)^2))

# Define a function to get predictions using bagging
bagging_prediction <- function(train_data, test_data, B = 100) {
  predictions <- numeric(nrow(test_data))
  
  for(b in 1:B) {
    # Create a bootstrap sample of the training data
    indices <- sample(1:nrow(train_data), replace = TRUE)
    bootstrap_data <- train_data[indices, ]
    
    # Train a decision tree on the bootstrap data
    model <- rpart(quality ~ ., data=bootstrap_data, method="anova")
    predictions <- predictions + predict(model, newdata=test_data)
  }
  
  # Return the average prediction
  return(predictions / B)
}

# Apply bagging for predictions
predictionsBagRed <- bagging_prediction(trainSetRed, validSetRed)
predictionsBagWhite <- bagging_prediction(trainSetWhite, validSetWhite)

# Compute RMSE for Bagging
rmseBagRed <- sqrt(mean((predictionsBagRed - validSetRed$quality)^2))
rmseBagWhite <- sqrt(mean((predictionsBagWhite - validSetWhite$quality)^2))

# Boosting for Red Wine
boostModelRed <- gbm(quality ~ ., distribution = "gaussian", data = trainSetRed, n.trees = 5000, interaction.depth = 6, shrinkage = 0.005)
summary(boostModelRed)

# Predict on validation set for Red Wine
predictionsBoostRed <- predict(boostModelRed, newdata = validSetRed, n.trees = 5000)
rmseBoostRed <- sqrt(mean((predictionsBoostRed - validSetRed$quality)^2))

# Boosting for White Wine
boostModelWhite <- gbm(quality ~ ., distribution = "gaussian", data = trainSetWhite, n.trees = 5000, interaction.depth = 6, shrinkage = 0.005)
summary(boostModelWhite)


# Predict on validation set for White Wine
predictionsBoostWhite <- predict(boostModelWhite, newdata = validSetWhite, n.trees = 5000)
rmseBoostWhite <- sqrt(mean((predictionsBoostWhite - validSetWhite$quality)^2))

# Print RMSE for Boosting
cat("RMSE for Boosting on Red Wine:", rmseBoostRed, "\n")
cat("RMSE for Boosting on White Wine:", rmseBoostWhite, "\n")


# Print RMSE
cat("RMSE for Bagging on Red Wine:", rmseBagRed, "\n")
cat("RMSE for Bagging on White Wine:", rmseBagWhite, "\n")

# Print the RMSE values
cat("RMSE for RandomForest on Red Wine:", rmseRfRed, "\n")
cat("RMSE for RandomForest on White Wine:", rmseRfWhite, "\n")
cat("RMSE for XGBoost on Red Wine:", rmseXgbRed, "\n")
cat("RMSE for XGBoost on White Wine:", rmseXgbWhite, "\n")

# Predict with RandomForest models on the test data
predictionsRfRed <- predict(rfModelRed, newdata=testRed[,-1]) # -1 to exclude the Id column
predictionsRfWhite <- predict(rfModelWhite, newdata=testWhite[,-1])

# Create the submission dataframe
submissionRfRed <- data.frame(Id = testRed$Id, quality = predictionsRfRed)
submissionRfWhite <- data.frame(Id = testWhite$Id, quality = predictionsRfWhite)
submission <- rbind(submissionRfRed, submissionRfWhite)



write.csv(x = submission, file = "project/volume/data/processed/submission.csv", row.names = F)

