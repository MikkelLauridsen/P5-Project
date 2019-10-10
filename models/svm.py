import os
from sklearn.svm import SVC

import datareader_csv
from models.model_utility import scale_features
from models.model_utility import split_feature_label
from models.model_utility import find_best_hyperparameters

# Going up one directory so we have access to the below specified files.
os.chdir("..")

training_filepath = "data/idpoint_dataset/mixed_training_67761_100ms.csv"
test_filepath = "data/idpoint_dataset/mixed_test_14521_100ms.csv"

X_train, y_train = split_feature_label(datareader_csv.load_idpoints(training_filepath))
X_test, y_test = split_feature_label(datareader_csv.load_idpoints(test_filepath))

X_train, X_test = scale_features(X_train, X_test)

# Setting up a parameter grid to define the different hyperparameters and their ranges.
param_grid = [
  {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
  {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf', "poly", "sigmoid"]},
 ]

find_best_hyperparameters(SVC(), param_grid, X_train, y_train, X_test, y_test)
