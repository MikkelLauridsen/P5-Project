from sklearn.neural_network import MLPClassifier
import os

import datareader_csv
from models.model_utility import find_best_hyperparameters
from models.model_utility import scale_features
from models.model_utility import split_feature_label

# Going up one directory so we have access to the below specified files.
os.chdir("..")

# load training idpoints
# training_points = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_135520_100ms.csv", 0)
# test_points = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_29041_100ms.csv", 0)

training_points = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_77441_100ms.csv", 0)
test_points = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_19361_100ms.csv", 0)

X_train, y_train = split_feature_label(training_points)
X_test, y_test = split_feature_label(test_points)
X_train, X_test = scale_features(X_train, X_test)

parameter_space = [{'solver': ['lbfgs'],
                    'alpha': [1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06],
                    'hidden_layer_sizes': [(16, 2), (12, 3)],
                    'random_state': [1]}]

find_best_hyperparameters(MLPClassifier(), parameter_space, X_train, y_train, X_test, y_test)
