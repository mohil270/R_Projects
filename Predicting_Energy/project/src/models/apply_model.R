  rm(list = ls())
  
  # Load required libraries
  library(dplyr)
  library(glmnet)
  library(Matrix)
  
  # Load the test data
  test_data <- read.csv("./project/volume/data/raw/test.csv")
  
  test_data$X <- NULL
  column_means <- colMeans(test_data, na.rm = TRUE)
  for (col in names(test_data)) {
    test_data[is.na(test_data[, col]), col] <- column_means[col]
  }
  
  
  # Convert test data to matrix
  test_X <- sparse.model.matrix(~., data = test_data)
  
  # Load the trained models
  lasso_final_model <- readRDS("./project/volume/models/lasso_final_model.rds")
  ridge_final_model <- readRDS("./project/volume/models/ridge_final_model.rds")
  
  # Predict on the test set
  lasso_preds <- predict(lasso_final_model, newx = test_X)
  ridge_preds <- predict(ridge_final_model, newx = test_X)
  
  # Create a submission data frame with Id and Energ_Kcal columns
  submission <- data.frame(Id = test_data$Id, Energ_Kcal = lasso_preds)
  
  colnames(submission)[colnames(submission) == "s0"] <- "Energ_Kcal"
  
  # Save the submission file
  write.csv(submission, "./project/volume/data/processed/submission_lasso_file.csv", row.names = FALSE)
  
