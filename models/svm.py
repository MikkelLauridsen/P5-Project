import os
from sklearn.svm import SVC

import datareader_csv
from models.model_utility import scale_features, find_best_hyperparameters2
from models.model_utility import split_feature_label
from models.model_utility import find_best_hyperparameters

if __name__ == "__main__":
    # Going up one directory so we have access to the below specified files.
    os.chdir("..")

    training_filepath = "data/idpoint_dataset/mixed_training_77441_100ms.csv"
    test_filepath = "data/idpoint_dataset/mixed_test_19361_100ms.csv"

    X_train, y_train = split_feature_label(datareader_csv.load_idpoints(training_filepath))
    X_test, y_test = split_feature_label(datareader_csv.load_idpoints(test_filepath))

    X_train, X_test = scale_features(X_train, X_test)

    # Setting up a parameter grid to define the different hyperparameters and their ranges.
    param_grid = [
      {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
      {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf', "poly", "sigmoid"]},
     ]

    find_best_hyperparameters(SVC(), param_grid, X_train, y_train, X_test, y_test)

def svm(X_train, y_train):
    # set up the parameter grid
    print("Grid for SVM is now being set up")
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
        {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf', "poly", "sigmoid"]},
    ]
    print("Grid for SVM has now been set up, hyper parameters are being found")

    #Find hyperparameters
    svm_model = find_best_hyperparameters2(SVC(), param_grid, X_train, y_train)

    # Model gets send back to get accuracy taken
    print("The SVM model has now been created, and prediction of accuracy is now being calculated")
    return svm_model