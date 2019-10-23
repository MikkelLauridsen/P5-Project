from sklearn.ensemble import AdaBoostClassifier
import os

import datareader_csv
from models.model_utility import find_best_hyperparameters, best_hyper_parameters_for_all_model
from models.model_utility import scale_features
from models.model_utility import split_feature_label

# Going up one directory so we have access to the below specified files.
if __name__ == "__main__":
    os.chdir("..")

    # Single Decision Tree
    training_data = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_77441_100ms.csv", 0)
    test_data = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_19361_100ms.csv", 0)

    X_train, y_train = split_feature_label(training_data)
    X_test, y_test = split_feature_label(test_data)
    X_train, X_test = scale_features(X_train, X_test)

    # Max-Depth of tree, minimum amount of samples per node, max amount of features to consider, and impurity measurement
    parameter_grid = [{'n_estimators': [3, 5, 8, 9, 10, 11, 12, 13],
                       'algorithm': ['SAMME.R', 'SAMME'],
                       'learning_rate': [0.1, 0.5, 1.0]}]

    find_best_hyperparameters(AdaBoostClassifier(), parameter_grid, X_train, y_train, X_test, y_test)


def ada_boost(X_train, y_train):
    # Grid is set up
    print("Grid for ada boost is now being set up")
    parameter_grid = [{'n_estimators': [3, 5, 8, 9, 10, 11, 12, 13],
                       'algorithm': ['SAMME.R', 'SAMME'],
                       'learning_rate': [0.1, 0.5, 1.0]}]

    # Hyper parameters found
    print("Grid for ada boost has now been set up, hyper parameters are being found")
    ada_boost_model = best_hyper_parameters_for_all_model(AdaBoostClassifier(), parameter_grid, X_train, y_train)

    # Model is send back to have accuracy predicted
    print("The ada boost model has now been created, and prediction of accuracy is now being calculated")
    return ada_boost_model
