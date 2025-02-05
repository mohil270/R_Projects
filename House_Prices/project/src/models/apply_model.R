rm(list = ls())

# Load required libraries
library(dplyr)
library(glmnet)
library(Matrix)

# Load the test data
test_data <- read.csv("./project/volume/data/raw/Stat_380_test.csv")

# Handle missing values
test_data$LotFrontage[is.na(test_data$LotFrontage)] <- mean(test_data$LotFrontage, na.rm = TRUE)
test_data$Heating[is.na(test_data$Heating)] <- "GasA" # Impute with mode
test_data$CentralAir[is.na(test_data$CentralAir)] <- "Y" # Impute with mode

# Perform feature engineering if needed
current_year <- as.numeric(format(Sys.Date(), "%Y"))
test_data$Age <- current_year - test_data$YearBuilt

# Convert categorical variables to numerical representation
test_data <- test_data %>%
  mutate(BldgType_1Fam = ifelse(BldgType == "1Fam", 1, 0),
         BldgType_TwnhsE = ifelse(BldgType == "TwnhsE", 1, 0),
         Heating_GasA = ifelse(Heating == "GasA", 1, 0),
         Heating_GasW = ifelse(Heating == "GasW", 1, 0),
         CentralAir_Y = ifelse(CentralAir == "Y", 1, 0))

# Save the preprocessed test data
write.csv(test_data, "./project/volume/data/processed/preprocessed_test_data.csv", row.names = FALSE)

# Convert test data to matrix
test_X <- sparse.model.matrix(~., data = test_data)

# Load the trained models
lasso_final_model <- readRDS("./project/volume/models/lasso_final_model.rds")
ridge_final_model <- readRDS("./project/volume/models/ridge_final_model.rds")

# Predict on the test set
lasso_preds <- predict(lasso_final_model, newx = test_X)
ridge_preds <- predict(ridge_final_model, newx = test_X)

# Create a submission data frame with Id and SalePrice columns
submission <- data.frame(Id = test_data$Id, SalePrice = ridge_preds)

colnames(submission)[colnames(submission) == "s0"] <- "SalePrice"

# Save the submission file
write.csv(submission, "./project/volume/data/processed/submission_RIDGE_file.csv", row.names = FALSE)

