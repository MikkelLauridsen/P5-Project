import datapoint
import os
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from datasets import load_or_create_datasets
from models.decision_trees import dt
from models.knn import knn
from models.logistic_regression import lr
from models.mlp import mlp
from models.nbc import nbc
from models.random_forest import rf
from models.svm import svm

# TODO: bayesian networks!


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
def find_best_hyperparameters(estimator, parameter_grid, X_train, y_train, X_test, y_test):
    # Creating the grid search that uses the parameter grid to find the best hyperparameters.
    grid_s = GridSearchCV(estimator, parameter_grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=10)
    grid_s.fit(X_train, y_train)

    print(f"parameters found: {grid_s.best_params_}")

    y_predict = grid_s.predict(X_test)

    return y_predict


def best_hyper_parameters_for_all_model(estimator, parameter_grid, X_train, y_train):
    # Creating the grid search that uses the parameter grid to find the best hyperparameters.
    grid_s = GridSearchCV(estimator, parameter_grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=10)
    grid_s.fit(X_train, y_train)

    print(f"parameters found: {grid_s.best_params_}")
    return grid_s


def print_metrics(metric_dic):
    print("printing classification metrics:")

    labels = ['Precision', 'Recall', 'TNR', 'FPR', 'FNR', 'Balanced accuracy', 'F1-score']
    print("{:15}".format(" ") + "".join(["{:>18}".format(f"{label} ") for label in labels]))

    for key in metric_dic.keys():
        line = "{:>15}".format(f"{key}  ")

        for i in range(len(labels)):
            line += "{:17.4f}".format(metric_dic[key][i]) + " "

        print(line)


def save_metrics(metric_dic, period_ms, shuffle, stride_ms, impersonation_split, dos_type, model, threshold, parameters={}):
    path, dir = __get_metrics_path(period_ms, shuffle, stride_ms, impersonation_split, dos_type, model, threshold, parameters)
    labels = ['Precision', 'Recall', 'TNR', 'FPR', 'FNR', 'Balanced accuracy', 'F1-score']

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(['Class'] + labels)

        for key in metric_dic.keys():
            writer.writerow([key] + list(metric_dic[key]))


def load_metrics(period_ms, shuffle, stride_ms, impersonation_split, dos_type, model, threshold, parameters={}):
    path, _ = __get_metrics_path(period_ms, shuffle, stride_ms, impersonation_split, dos_type, model, threshold, parameters)
    metric_dic = {}

    with open(path, newline="") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader, None)

        for row in reader:
            metric_dic[row[0]] = [float(string) for string in row[1:]]

    return metric_dic


def get_classifier(model, X, y, parameters={}):
    if model == 'mlp':
        return mlp(X, y) if len(parameters.keys()) == 0 else mlp(X, y, parameters)
    elif model == 'knn':
        return knn(X, y) if len(parameters.keys()) == 0 else knn(X, y, parameters)
    elif model == 'svm':
        return svm(X, y) if len(parameters.keys()) == 0 else svm(X, y, parameters)
    elif model == 'rf':
        return rf(X, y) if len(parameters.keys()) == 0 else rf(X, y, parameters)
    elif model == 'nbc':
        return nbc(X, y) if len(parameters.keys()) == 0 else nbc(X, y, parameters)
    elif model == 'lr':
        return lr(X, y) if len(parameters.keys()) == 0 else lr(X, y, parameters)
    elif model == 'dt':
        return dt(X, y) if len(parameters.keys()) == 0 else dt(X, y, parameters)
    # elif model == 'bn': TODO: add bayesian networks
    #    return bn(X, y, parameters)
    else:
        raise ValueError()


def load_or_create_metrics(period_ms, shuffle, stride_ms, impersonation_split, dos_type, model, threshold, parameters={}):
    path, _ = __get_metrics_path(period_ms, shuffle, stride_ms, impersonation_split, dos_type, model, threshold, parameters)

    if os.path.exists(path):
        return load_metrics(period_ms, shuffle, stride_ms, impersonation_split, dos_type, model, threshold, parameters)

    training_data, test_data = load_or_create_datasets(period_ms, shuffle, stride_ms, impersonation_split, dos_type)
    X_train, y_train = split_feature_label(training_data)
    X_test, y_test = split_feature_label(test_data)
    X_train, X_test = scale_features(X_train, X_test)

    X_train, X_test = select_features(X_train, X_test, threshold)

    classifier = get_classifier(model, X_train, y_train, parameters)  # TODO: cross validation
    y_predict = classifier.predict(X_test)
    metric_dic = get_metrics(y_test, y_predict)

    save_metrics(metric_dic, period_ms, shuffle, stride_ms, impersonation_split, dos_type, model, threshold, parameters)

    return metric_dic


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
    tp = fp = tn = fn = 0
    class_counters = {'normal': [0, 0, 0, 0], 'dos': [0, 0, 0, 0], 'fuzzy': [0, 0, 0, 0], 'impersonation': [0, 0, 0, 0]}

    if len(y_test) != len(y_predict):
        raise IndexError()

    for i in range(len(y_test)):
        if y_test[i] == y_predict[i]:
            __increment_equal(y_test[i], class_counters)

            if y_test[i] == 'normal':
                tn += 1
            else:
                tp += 1
        else:
            __increment_not_equal(y_test[i], y_predict[i], class_counters)

            if y_test[i] == 'normal':
                fn += 1
            else:
                fp += 1

    metrics = {'total': __get_metrics_tuple(tp, fp, tn, fn)}

    for key in class_counters.keys():
        counts = class_counters[key]
        metrics[key] = __get_metrics_tuple(counts[0], counts[1], counts[2], counts[3])

    return metrics


# Returns the file and directory paths associated with input argument combination.
def __get_metrics_path(period_ms, shuffle, stride_ms, impersonation_split, dos_type, model, threshold, parameters={}):
    imp_name = "imp_split" if impersonation_split else "imp_full"
    baseline_name = "baseline" if len(parameters.keys()) == 0 else "modified"
    shuffle_name = "shuffled" if shuffle else "normal"
    name = f"mixed_{period_ms}ms_{stride_ms}ms_{shuffle_name}_{threshold}"

    for parameter in parameters:
        name += f"_{parameter}"

    dir = f"result/{baseline_name}/{model}/{imp_name}/{dos_type}/"

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
    precision = 1 if tp + fp == 0 else tp / (tp + fp)
    recall = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    balanced_accuracy = (recall + tnr) / 2
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, tnr, fpr, fnr, balanced_accuracy, f1


if __name__ == "__main__":
    os.chdir("..")
    print_metrics(load_or_create_metrics(100, True, 100, False, 'modified', 'mlp', 0.1, parameters={'hidden_layer_sizes': (12, 2)}))