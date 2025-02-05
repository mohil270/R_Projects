rm(list = ls())

# Load required libraries
library(dplyr)
library(glmnet)
library(Matrix)

# Load the test data
test_data <- read.csv("./project/volume/data/raw/test.csv")

# Remove the Id column from the test data as it's not needed for predictions
test_data <- test_data %>%
  select(-Id)

# Fill missing values in the test data with the corresponding column means
column_means <- colMeans(test_data, na.rm = TRUE)
for (col in names(test_data)) {
  test_data[is.na(test_data[, col]), col] <- column_means[col]
}

# Convert test data to matrix
test_X <- sparse.model.matrix(~., data = test_data)

# Load the trained Lasso model
lasso_final_model <- readRDS("./project/volume/models/lasso_final_model.rds")

ridge_final_model <- readRDS("./project/volume/models/ridge_final_model.rds")
# Predict on the test set using the Lasso model
lasso_preds <- predict(lasso_final_model, newx = test_X, s = "lambda.min", type = "response")
binary_preds <- as.integer(ridge_preds >= 0.5)  # Convert probabilities to binary predictions (0 or 1)

# Create a submission data frame with Id and predicted Signal columns
submission <- data.frame(Id = 1:nrow(test_data), Signal = binary_preds)

# Save the submission file
write.csv(submission, "./project/volume/data/processed/submission_binary_file.csv", row.names = FALSE)
