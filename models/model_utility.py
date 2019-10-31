import datapoint
import os
import csv
import time
import pandas as pd
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


# Finds the best combination of hyperparameters and prints the results.
def find_best_hyperparameters(estimator, parameter_grid, X_train, y_train, X_test):
    # Creating the grid search that uses the parameter grid to find the best hyperparameters.
    grid_s = GridSearchCV(estimator, parameter_grid, cv=5, n_jobs=-1, scoring="f1_macro")
    grid_s.fit(X_train, y_train)

    before = time.perf_counter_ns()
    y_predict = grid_s.predict(X_test)
    time_ns = time.perf_counter_ns() - before

    return y_predict, time_ns


def print_metrics(metrics):
    print("printing classification metrics:")

    labels = ['Precision', 'Recall', 'TNR', 'FPR', 'FNR', 'Balanced accuracy', 'F1-score']
    print("{:15}".format(" ") + "".join(["{:>18}".format(f"{label} ") for label in labels]))

    for key in metrics.keys():
        line = "{:>15}".format(f"{key}  ")

        for i in range(len(labels)):
            line += "{:17.4f}".format(metrics[key][i]) + " "

        print(line)


def save_metrics(metrics, period_ms, stride_ms, imp_split, dos_type, model, parameters, subset):
    path, dir = get_metrics_path(period_ms, stride_ms, imp_split, dos_type, model, parameters, subset)
    labels = ['Precision', 'Recall', 'TNR', 'FPR', 'FNR', 'Balanced accuracy', 'F1-score']

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(['Class'] + labels)

        for key in metrics.keys():
            writer.writerow([key] + list(metrics[key]))


def save_time(time_model, time_feature, period_ms, overlap_ms, imp_split, dos_type, model, parameters, subset):
    path, dir = get_metrics_path(period_ms, overlap_ms, imp_split, dos_type, model, parameters, subset, True)

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(['Model_time', 'Feature_time', 'Total'])
        writer.writerow([time_model, time_feature, time_model + time_feature])


def load_metrics(period_ms, stride_ms, imp_split, dos_type, model, parameters, subset):
    path, _ = get_metrics_path(period_ms, stride_ms, imp_split, dos_type, model, parameters, subset)
    metrics = {}

    with open(path, newline="") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader, None)

        for row in reader:
            metrics[row[0]] = [float(string) for string in row[1:]]

    return metrics


def get_classifier(model):
    if model == 'mlp':
        return mlp()
    elif model == 'knn':
        return knn()
    elif model == 'svm':
        return svm()
    elif model == 'rf':
        return rf()
    elif model == 'nbc':
        return nbc()
    elif model == 'lr':
        return lr()
    elif model == 'dt':
        return dt()
    elif model == 'bn':
        return bn()
    else:
        raise ValueError()


def select_features(X_train, X_test, threshold):
    column_labels = list(datapoint.datapoint_attributes)[2:]

    # get correlations of each feature in dataset
    data = pd.DataFrame(X_train, columns=column_labels)
    corrmat = data.corr(method='spearman')
    feature_indices = set()
    eliminated = set()

    for label in column_labels:
        features = abs(corrmat[label])
        exceeds = features[features >= threshold].to_dict()

        for key in exceeds.keys():
            if key != label and key not in eliminated:
                eliminated.add(label)
                break

    for label in eliminated:
        feature_indices.add(column_labels.index(label))

    X_train = __strip_features(X_train, feature_indices)
    X_test = __strip_features(X_test, feature_indices)

    return X_train, X_test


def __strip_features(X, feature_indices):
    feature_count = len(list(datapoint.datapoint_attributes)[2:])
    X_mod = []

    for i in range(len(X)):
        sample = list(X[i])

        for j in reversed(range(0, feature_count)):
            if j in feature_indices and len(sample) > 1:
                del sample[j]

        X_mod.append(sample)

    return X_mod


def get_metrics(y_test, y_predict):
    class_counters = {'normal': [0, 0, 0, 0], 'dos': [0, 0, 0, 0], 'fuzzy': [0, 0, 0, 0], 'impersonation': [0, 0, 0, 0]}

    if len(y_test) != len(y_predict):
        raise IndexError()

    for i in range(len(y_test)):
        if y_test[i] == y_predict[i]:
            __increment_equal(y_test[i], class_counters)
        else:
            __increment_not_equal(y_test[i], y_predict[i], class_counters)

    metrics = {}

    for key in class_counters.keys():
        counts = class_counters[key]
        metrics[key] = __get_metrics_tuple(counts[0], counts[1], counts[2], counts[3])

    metrics['total'] = __get_samples_tuple(metrics)

    return metrics


# Returns the file and directory paths associated with input argument combination.
def get_metrics_path(period_ms, stride_ms, imp_split, dos_type, model, parameters, subset, is_time=False):
    imp_name = "imp_split" if imp_split else "imp_full"
    baseline_name = "baseline" if len(parameters.keys()) == 0 else "selected_parameters"
    name = f"mixed_{period_ms}ms_{stride_ms}ms"
    result = "result" if is_time == False else "time"

    labels = list(datapoint.datapoint_attributes)[2:]

    for label in subset:
        name += f"_{labels.index(label)}"

    for parameter in parameters:
        name += f"_{parameter}"

    dir = f"{result}/{baseline_name}/{model}/{imp_name}/{dos_type}/"

    return dir + name + ".csv", dir


def __increment_equal(label, class_counters):
    class_counters[label][0] += 1

    for key in class_counters.keys():
        if key != label:
            class_counters[key][2] += 1


def __increment_not_equal(label, prediction, class_counters):
    class_counters[label][3] += 1
    class_counters[prediction][1] += 1

    for key in class_counters.keys():
        if key != label and key != prediction:
            class_counters[key][2] += 1


def __get_metrics_tuple(tp, fp, tn, fn):
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1 if tp + fn == 0 else tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    balanced_accuracy = (recall + tnr) / 2
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return precision, recall, tnr, fpr, fnr, balanced_accuracy, f1


def __get_samples_tuple(metrics):
    micros = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for label in metrics.keys():
        micros = list(map(add, micros, metrics[label]))

    length = len(metrics.keys())

    return [metric / length for metric in micros]


def get_standard_feature_split():
    """Returns the feature split used for finding hyperparameters in the standard case."""
    # Loading the standard dataset
    data_train, data_test, _ = datasets.load_or_create_datasets(period_ms=50, stride_ms=50, impersonation_split=False,
                                                                dos_type="modified")

    # Splitting the data into features and labels.
    X_train, y_train = split_feature_label(data_train)
    X_test, y_test = split_feature_label(data_test)

    # Returning the scaled versions.
    X_train, _ = scale_features(X_train, X_test)

    return X_train, y_train
