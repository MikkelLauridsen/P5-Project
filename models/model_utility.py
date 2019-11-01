import datapoint
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from models.decision_trees import dt
from models.knn import knn
from models.logistic_regression import lr
from models.mlp import mlp
from models.nbc import nbc
from models.random_forest import rf
from models.svm import svm
from operator import add
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


# Fitting a transformation on the training features and scaling both the training features and test features to it.
def scale_features(X_training, X_test):
    # fit the scaler
    scaler = StandardScaler()
    scaler.fit(X_training)

    # Transforming the training and test features with the fitted scaler.
    X_training = scaler.transform(X_training)
    X_test = scaler.transform(X_test)

    return X_training, X_test


def find_best_hyperparameters(estimator, parameter_grid, X_train, y_train, X_test):
    """Conducts a gridsearch of specified model with combinations of parameters in specified space.
    Returns:
        - y_predict: a list of predicted class labels
        - time_ns:   the time spent predicting labels (int ns)
    Parameters are:
        - estimator:      the classifier
        - parameter_grid: the model's parameter space {'**parameter**': [**values**]}
        - X_train:        a list of training feature values
        - y_train:        a list of training class labels
        - X_test:         a list of test feature values
    """

    # Creating the grid search that uses the parameter grid to find the best hyperparameters.
    grid_s = GridSearchCV(estimator, parameter_grid, cv=5, n_jobs=-1, scoring="f1_macro", verbose=10)
    grid_s.fit(X_train, y_train)

    before = time.perf_counter_ns()
    y_predict = grid_s.predict(X_test)
    time_ns = (time.perf_counter_ns() - before) / len(X_test)

    return y_predict, time_ns


def print_metrics(metrics):
    """Outputs a classification report to console, based on specified metrics"""

    print("printing classification metrics:")

    labels = ['Precision', 'Recall', 'TNR', 'FPR', 'FNR', 'Balanced accuracy', 'F1-score']
    print("{:15}".format(" ") + "".join(["{:>18}".format(f"{label} ") for label in labels]))

    for key in metrics.keys():
        line = "{:>15}".format(f"{key}  ")

        for i in range(len(labels)):
            line += "{:17.4f}".format(metrics[key][i]) + " "

        print(line)


def get_classifier(model):
    """returns a classification model based on specified model name,
    which may be one of ('mlp', 'knn', 'svm', 'rf', 'nbc', 'lr', 'dt', 'bn')"""

    classifier = {
        'mlp': mlp(),
        'knn': knn(),
        'svm': svm(),
        'rf':  rf(),
        'nbc': nbc(),
        'lr': lr(),
        'dt': dt(),
        'bn': bn()
    }.get(model, None)

    if classifier is None:
        raise ValueError()
    else:
        return classifier


def get_metrics(y_test, y_predict):
    """returns a dictionary of metrics of the form:
        - {'**class**': (precision, recall, tnr, fpr, fnr, balanced_accuracy, f1)}
    Parameters are:
        - y_test:    a list of actual class labels
        - y_predict: a list of predicted class labels"""

    if len(y_test) != len(y_predict):
        raise IndexError()

    class_counters = {
        'normal': [0, 0, 0, 0],
        'dos': [0, 0, 0, 0],
        'fuzzy': [0, 0, 0, 0],
        'impersonation': [0, 0, 0, 0]}

    for i in range(len(y_test)):
        if y_test[i] == y_predict[i]:
            __increment_equal(y_test[i], class_counters)
        else:
            __increment_not_equal(y_test[i], y_predict[i], class_counters)

    metrics = {}

    for key in class_counters.keys():
        counts = class_counters[key]
        metrics[key] = __get_metrics_tuple(counts[0], counts[1], counts[2], counts[3])

    metrics['total'] = __get_macro_tuple(metrics)

    return metrics


def get_metrics_path(period_ms, stride_ms, imp_split, dos_type, model, parameters, subset, is_time=False):
    """returns the file path and directory path associated with the specified parameters.
    Parameters are:
        - period_ms:  window size (int ms)
        - stride_ms:  stride size (int ms)
        - imp_split:  the impersonation type (True, False)
        - dos_type:   the DoS type ('modified', 'original')
        - model:      model name ('bn', 'dt', 'knn', 'lr', 'mlp', 'nbc', 'rf', 'svm')
        - parameters: model parameter space {'**parameter**': [**values**]}
        - subset:     a list of labels of features to be used
        - is_time:    whether the path is related to scores or durations (True, False)"""

    imp_name = "imp_split" if imp_split else "imp_full"
    baseline_name = "baseline" if len(parameters.keys()) == 0 else "selected_parameters"
    metric_type = "score" if is_time is False else "time"
    name = f"mixed_{metric_type}_{period_ms}ms_{stride_ms}ms"

    labels = list(datapoint.datapoint_attributes)[2:]

    for label in subset:
        name += f"_{labels.index(label)}"

    dir = f"result/{baseline_name}/{model}/{imp_name}/{dos_type}/"

    return dir + name + ".csv", dir


def get_dataset(period_ms, stride_ms, imp_split, dos_type):
    """returns the scaled and split equivalent of the dataset associated with specified parameters:
        - X_train:           feature values of the training set
        - y_train:           class labels of the training set
        - X_test:            feature values of the test set
        - y_test:            class labels of the test set
        - feature_time_dict: a dictionary of {'**feature**': **time_ns**}

    If it does not exist, it is created.
    Parameters are:
        - period_ms: the window size (int ms)
        - stride_ms: the stride size (int ms)
        - imp_split: the impersonation type (True, False)
        - dos_type:  the DoS type ('modified', 'original')"""
    training_data, test_data, feature_time_dict = datasets.load_or_create_datasets(
        period_ms,
        True,
        stride_ms,
        imp_split,
        dos_type,
        verbose=True)

    X_train, y_train = split_feature_label(training_data)
    X_test, y_test = split_feature_label(test_data)
    X_train, X_test = scale_features(X_train, X_test)

    return X_train, y_train, X_test, y_test, feature_time_dict


def get_standard_feature_split():
    """Returns the feature split used for finding hyperparameter values in the standard case."""
    # Loading the standard dataset
    data_train, data_test, _ = datasets.load_or_create_datasets(
        period_ms=50,
        stride_ms=50,
        impersonation_split=False,
        dos_type="modified")

    # Splitting the data into features and labels.
    X_train, y_train = split_feature_label(data_train)
    X_test, y_test = split_feature_label(data_test)

    # Returning the scaled versions.
    X_train, _ = scale_features(X_train, X_test)

    return X_train, y_train


# updates specified class counters dictionary based on specified label
def __increment_equal(label, class_counters):
    class_counters[label][0] += 1

    for key in class_counters.keys():
        if key != label:
            class_counters[key][2] += 1


# updates specified class counters dictionary based on actual and predicted labels, which are different
def __increment_not_equal(label, prediction, class_counters):
    class_counters[label][3] += 1
    class_counters[prediction][1] += 1

    for key in class_counters.keys():
        if key != label and key != prediction:
            class_counters[key][2] += 1


# returns a list of evaluation metrics based on specified classification counters
def __get_metrics_tuple(tp, fp, tn, fn):
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1 if tp + fn == 0 else tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    balanced_accuracy = (recall + tnr) / 2
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return precision, recall, tnr, fpr, fnr, balanced_accuracy, f1


# finds the macro average of class scores in specified metric dictionary
def __get_macro_tuple(metrics):
    micros = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for label in metrics.keys():
        micros = list(map(add, micros, metrics[label]))

    length = len(metrics.keys())

    return [metric / length for metric in micros]
