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


def split_feature_label(datapoints):
    """Splits specified list of datapoints into two lists: a list of lists of feature values, a list of class labels."""
    X = []  # The list of lists of feature values
    y = []  # The list of class labels

    # Going through each instance and extracting the features and labels
    for point in datapoints:
        features = []
        for attr in datapoint.datapoint_attributes:
            if attr == "time_ms":
                pass
            elif attr == "class_label":
                y.append(point.class_label)
            else:
                features.append(getattr(point, attr))

        X.append(features)

    return X, y


def scale_features(X_training, X_validation):
    """Returns scaled equivalents of specified training and validation feature values,
    such that mean is 0 and variance is 1."""
    # Fit the scaler to the training features
    scaler = StandardScaler()
    scaler.fit(X_training)

    # Transform the training and validation features with the fitted scaler
    X_training = scaler.transform(X_training)
    X_validation = scaler.transform(X_validation)

    return X_training, X_validation


def find_best_hyperparameters(estimator, parameter_grid, X_train, y_train):
    """Conducts a gridsearch of specified model with combinations of parameters in specified space,
    and returns the best combination of parameter values.

    :param estimator: the baseline classifier to be used.
    :param parameter_grid: a dictionary from feature names to lists of values {'**parameter**': [**values**]}.
    :param X_train: a list of lists of feature values.
    :param y_train: a list of class labels.
    :return: a dictionary of the best parameters {'**parameter**': **value**}.
    """

    # Conduct the grid search that uses the parameter grid to find the best hyperparameters
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
    """Returns the scaled and split equivalent of the dataset associated with specified parameters.
    If the dataset does not exist, it is created.

    :param period_ms: the window size (int ms).
    :param stride_ms: the stride size (int ms).
    :param imp_split: the impersonation type (True, False).
    :param dos_type:  the DoS type ('modified', 'original').
    :return: scaled feature value list, class label list, a dictionary of feature times {'**feature**': **time_ns**}.
    """
    test_data, feature_time_dict = datasets.get_mixed_test(period_ms, stride_ms, imp_split, dos_type, verbose=True)
    X_test, y_test = split_feature_label(test_data)

    scaler = StandardScaler()
    scaler.fit(X_test)

    return scaler.transform(X_test), y_test, feature_time_dict


def get_scaled_training_validation(period_ms, stride_ms, imp_split, dos_type):
    """Returns the scaled and split equivalent of the dataset associated with specified parameters.
    If it does not exist, it is created.

    :param period_ms: the window size (int ms).
    :param stride_ms: the stride size (int ms).
    :param imp_split: the impersonation type (True, False).
    :param dos_type:  the DoS type ('modified', 'original').
    :return: scaled training feature value list, training class label list, scaled validation feature list,
    validation class label list, dictionary of feature durations {'**feature**': **time_ns**}.
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
        period_ms=50, stride_ms=50,
        imp_split=False, dos_type='modified')

    # Split the data into features and labels
    X_train, y_train = split_feature_label(training_data)
    X_validation, y_validation = split_feature_label(validation_data)

    # Returning the scaled versions
    X_train, _ = scale_features(X_train, X_validation)

    return X_train, y_train
