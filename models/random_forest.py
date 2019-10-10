from sklearn.ensemble import RandomForestClassifier
import os

import datareader_csv
from models.model_utility import find_best_hyperparameters
from models.model_utility import scale_features
from models.model_utility import split_feature_label

# Going up one directory so we have access to the below specified files.
os.chdir("..")

# Single Decision Tree
training_data = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_135520_100ms.csv", 0)
test_data = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_29041_100ms.csv", 0)

X_train, y_train = split_feature_label(training_data)
X_test, y_test = split_feature_label(test_data)
X_train, X_test = scale_features(X_train, X_test)


# Max-Depth of tree, minimum amount of samples per node, max amount of features to consider, and impurity measurement
parameter_grid = [{'max_depth': [2, 5, 9],
                    'n_estimators': [3],
                    'min_samples_split': [2, 3, 5],
                    'max_features': [None, 2, 3],
                    'criterion': ['gini'],
                    'bootstrap': [True]}]

find_best_hyperparameters(RandomForestClassifier(), parameter_grid, X_train, y_train, X_test, y_test)
