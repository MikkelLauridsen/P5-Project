from sklearn.tree import DecisionTreeClassifier
import os
import datareader_csv
from models.model_utility import find_best_hyperparameters
from models.model_utility import scale_features
from models.model_utility import split_feature_label

# Going up one directory so we have access to the below specified files.
os.chdir("..")

# Single Decision Tree
training_data = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_77441_100ms.csv", 0)
test_data = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_19361_100ms.csv", 0)

X_train, y_train = split_feature_label(training_data)
X_test, y_test = split_feature_label(test_data)
X_train, X_test = scale_features(X_train, X_test)


# Max-Depth of tree, minimum amount of samples per node, max amount of features to consider, and impurity measurement
parameter_grid = [{'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    'min_samples_split': [2, 3, 5, 8],
                    'max_features': [None, 2, 3, 5],
                    'criterion': ['gini', 'entropy']}]

find_best_hyperparameters(DecisionTreeClassifier(), parameter_grid, X_train, y_train, X_test, y_test)