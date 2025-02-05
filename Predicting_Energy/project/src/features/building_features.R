rm(list = ls())

# Load the required libraries
library(caret)
library(dplyr)

# Read the train and test data
train_data <- read.csv("./project/volume/data/raw/train.csv")
test_data <- read.csv("./project/volume/data/raw/test.csv", skip = 1)

# Extract the first column from train_data and test_data
train_names <- train_data[[1]]
test_names <- test_data[[1]]

# Remove the first column from train_data and test_data
train_data <- train_data[, -1]
test_data <- test_data[, -1]

# Add Energ_Kcal column to test_data with NA values
test_data$Energ_Kcal <- NA

# Match column names of test_data to train_data
setnames(test_data, names(train_data))

# Combine the train and test data for feature engineering
combined_data <- rbindlist(list(train_data, test_data), use.names = TRUE)

# Replace missing values with column means
numeric_cols <- sapply(combined_data, is.numeric)
combined_data <- combined_data %>%
  mutate(across(.cols = names(combined_data)[numeric_cols], 
                .fns = ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Split the combined data back into train and test
new_train_data <- combined_data[1:nrow(train_data), ]

# Save the updated train and test data
fwrite(new_train_data, "./project/volume/data/processed/preprocessed_train_data.csv")

