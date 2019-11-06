import datapoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from models.decision_trees import dt
from models.knn import knn
from models.logistic_regression import lr
from models.mlp import mlp
from models.nbc import nbc
from models.random_forest import rf
from models.svm import svm
from models.bayesian_network import bn
import datasets


# Splitting the training data into feature and label lists.
def split_feature_label(datapoints):
    # The instance list that will contain all the features for each instance.
    X = []
    # The label list that will contain the injected status for each idpoint.
    y = []

    # Going through each instance and extracting the features and labels.
    for point in datapoints:
        features = []
        for attr in datapoint.datapoint_attributes:
            if attr == "time_ms":
                pass
            elif attr == "is_injected":
                y.append(point.is_injected)
            else:
                features.append(getattr(point, attr))

        X.append(features)

    return X, y


# Fitting a transformation on the training features,
# and scaling both the training features and validation features to it.
def scale_features(X_training, X_validation):
    # fit the scaler
    scaler = StandardScaler()
    scaler.fit(X_training)

    # Transforming the training and validation features with the fitted scaler.
    X_training = scaler.transform(X_training)
    X_validation = scaler.transform(X_validation)

    return X_training, X_validation


def find_best_hyperparameters(estimator, parameter_grid, X_train, y_train):
    """Conducts a gridsearch of specified model with combinations of parameters in specified space.
    :return:
        - a dictionary of the best parameters {'**parameter**': **value**}
    :parameter:
        - estimator:      the classifier
        - parameter_grid: the model's parameter space {'**parameter**': [**values**]}
        - X_train:        a list of training feature values
        - y_train:        a list of training class labels
    """

    # Creating the grid search that uses the parameter grid to find the best hyperparameters.
    grid_s = GridSearchCV(estimator, parameter_grid, cv=5, n_jobs=-1, scoring="f1_macro", verbose=10)
    grid_s.fit(X_train, y_train)

    return grid_s.best_params_


def get_classifier(model, parameters, subset):
    """returns a classification model based on specified model name,
    which may be one of ('mlp', 'knn', 'svm', 'rf', 'nbc', 'lr', 'dt', 'bn')"""

    if model == 'mlp':
        return mlp(parameters)
    elif model == 'knn':
        return knn(parameters)
    elif model == 'svm':
        return svm(parameters)
    elif model == 'rf':
        return rf(parameters)
    elif model == 'nbc':
        return nbc(parameters)
    elif model == 'lr':
        return lr(parameters)
    elif model == 'dt':
        return dt(parameters)
    elif model == 'bn':
        return bn(subset)
    else:
        raise ValueError


def get_scaled_test(period_ms, stride_ms, imp_split, dos_type):
    """returns the scaled and split equivalent of the dataset associated with specified parameters.
    :return:
        - X_test:           feature values of the test set
        - y_test:           class labels of the test set
        - feature_time_dict: a dictionary of {'**feature**': **time_ns**}
    :parameter:
        - period_ms: the window size (int ms)
        - stride_ms: the stride size (int ms)
        - imp_split: the impersonation type (True, False)
        - dos_type:  the DoS type ('modified', 'original')

    If it does not exist, it is created.
    """
    test_data, feature_time_dict = datasets.get_mixed_test(period_ms, stride_ms, imp_split, dos_type, verbose=True)
    X_test, y_test = split_feature_label(test_data)

    scaler = StandardScaler()
    scaler.fit(X_test)

    return scaler.transform(X_test), y_test, feature_time_dict


def get_scaled_training_validation(period_ms, stride_ms, imp_split, dos_type):
    """returns the scaled and split equivalent of the dataset associated with specified parameters.
    :return:
        - X_train:           feature values of the training set
        - y_train:           class labels of the training set
        - X_validation:            feature values of the validation set
        - y_validation:            class labels of the validation set
        - feature_time_dict: a dictionary of {'**feature**': **time_ns**}
    :parameter:
        - period_ms: the window size (int ms)
        - stride_ms: the stride size (int ms)
        - imp_split: the impersonation type (True, False)
        - dos_type:  the DoS type ('modified', 'original')

    If it does not exist, it is created.
    """
    training_data, validation_data, feature_time_dict = datasets.load_or_create_datasets(
        period_ms,
        stride_ms,
        imp_split,
        dos_type,
        verbose=True)

    X_train, y_train = split_feature_label(training_data)
    X_validation, y_validation = split_feature_label(validation_data)
    X_train, X_validation = scale_features(X_train, X_validation)

    return X_train, y_train, X_validation, y_validation, feature_time_dict


def get_standard_feature_split():
    """Returns the feature split used for finding hyperparameter values in the standard case."""
    # Loading the standard dataset
    training_data, validation_data, _ = datasets.load_or_create_datasets(
        period_ms=50,
        stride_ms=50,
        impersonation_split=False,
        dos_type="modified")

    # Splitting the data into features and labels.
    X_train, y_train = split_feature_label(training_data)
    X_validation, y_validation = split_feature_label(validation_data)

    # Returning the scaled versions.
    X_train, _ = scale_features(X_train, X_validation)

    return X_train, y_train
