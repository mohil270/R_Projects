rm(list = ls())
# Load required libraries
library(dplyr)

# Load the training data
train_data <- read.csv("./project/volume/data/raw/Stat_380_train.csv")

# Handlee missing values
train_data$LotFrontage[is.na(train_data$LotFrontage)] <- mean(train_data$LotFrontage, na.rm = TRUE)
train_data$Heating[is.na(train_data$Heating)] <- "GasA" # Impute with mode
train_data$CentralAir[is.na(train_data$CentralAir)] <- "Y" # Impute with mode

# Perform feature engineering if needed
# Example: create a new feature 'Age' by subtracting YearBuilt from the current year
current_year <- as.numeric(format(Sys.Date(), "%Y"))
train_data$Age <- current_year - train_data$YearBuilt

# Convertt categorical variables to numerical representation
# Example: using one-hot encoding
train_data <- train_data %>%
  mutate(BldgType_1Fam = ifelse(BldgType == "1Fam", 1, 0),
         BldgType_TwnhsE = ifelse(BldgType == "TwnhsE", 1, 0),
         Heating_GasA = ifelse(Heating == "GasA", 1, 0),
         Heating_GasW = ifelse(Heating == "GasW", 1, 0),
         CentralAir_Y = ifelse(CentralAir == "Y", 1, 0))

# Save the preprocessed data
write.csv(train_data, "./project/volume/data/processed/preprocessed_train_data.csv", row.names = FALSE)
