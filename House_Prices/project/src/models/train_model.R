rm(list = ls())

library(glmnet)
library(Matrix)

# Load thee preprocessed training data
train_data <- read.csv("./project/volume/data/processed/preprocessed_train_data.csv")

# Split the data into features (X) and target variable (y)
X <- train_data[, !(names(train_data) %in% "SalePrice")]
y <- train_data$SalePrice

# Convertt X to a sparse matrix
train_X <- sparse.model.matrix(~., data = X)

# Convert y to a numeric vector
train_y <- as.numeric(y)


# Train Lasso regression model
lasso_model <- cv.glmnet(train_X, train_y, alpha = 1) # alpha = 1 for Lasso

# Train Ridge regression model
ridge_model <- cv.glmnet(train_X, train_y, alpha = 0) # alpha = 0 for Ridge

# Select the best lambda value for each model
best_lambda_lasso <- lasso_model$lambda.min
best_lambda_ridge <- ridge_model$lambda.min

# Fit the models with the best lambda value
lasso_final_model <- glmnet(train_X, train_y, alpha = 1, lambda = best_lambda_lasso)
ridge_final_model <- glmnet(train_X, train_y, alpha = 0, lambda = best_lambda_ridge)

# Predict on the training set
lasso_preds <- predict(lasso_final_model, newx = train_X)
ridge_preds <- predict(ridge_final_model, newx = train_X)


# Compute RMSE for the predictions
rmse_lasso <- sqrt(mean((lasso_preds - train_y)^2))
rmse_ridge <- sqrt(mean((ridge_preds - train_y)^2))

cat("Lasso RMSE:", rmse_lasso, "\n")
cat("Ridge RMSE:", rmse_ridge, "\n")


# Save the trained models
saveRDS(lasso_final_model, file = "./project/volume/models/lasso_final_model.rds")
saveRDS(ridge_final_model, file = "./project/volume/models/ridge_final_model.rds")


