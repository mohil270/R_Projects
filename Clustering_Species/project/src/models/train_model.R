rm(list = ls())
library(cluster)

# read csv files
train <- read.csv("project/volume/data/raw/train.csv")
submission <- read.csv("project/volume/data/raw/submissionsample.csv")

# select only locus features for clustering
train_data <- train[, 2:16]  

# standardize the data
train_data <- scale(train_data)

set.seed(123)  # for reproducibility
clustering_results <- kmeans(train_data, centers=4)

train$cluster <- clustering_results$cluster

species1_cluster <- train$cluster[train$id == "sample_3"]
species2_cluster <- train$cluster[train$id == "sample_9"]
species3_cluster <- train$cluster[train$id == "sample_6"]

train$Species <- ifelse(train$cluster == species1_cluster, "species1", 
                        ifelse(train$cluster == species2_cluster, "species2", 
                               ifelse(train$cluster == species3_cluster, "species3", "species4")))

submission$Species <- train$Species
write.csv(x = submission, file = "project/volume/raw/submission.csv", row.names = F)
