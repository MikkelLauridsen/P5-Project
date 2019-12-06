import csv
import os
from metrics import get_metrics_path


def save_feature_durations(feature_durations, path, directory):
    """Saves specified feature duration dictionary to file at the specified path.

    :param feature_durations: a dictionary connecting feature names to durations {'**feature**': **duration int ns**}.
    :param path: the full file path.
    :param directory: the directory path.
    :return:
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(['Feature', 'Time'])

        for feature in feature_durations.keys():
            writer.writerow([feature, feature_durations[feature]])


def save_metrics(metrics, window_ms, stride_ms, imp_split, dos_type, model, parameters, subset, is_test=False):
    """Saves specified metrics to a file with the name associated with the remaining parameters.

    :param metrics: a dictionary of scores {'**class**': (precision, recall, tnr, fpr, fnr, balanced_accuracy, f1)}
    :param window_ms: the window size (int ms)
    :param stride_ms: the stride size (int ms)
    :param imp_split: the impersonation type (True, False)
    :param dos_type: the DoS type ('modified', 'original')
    :param model: the model name ('bn', 'dt', 'knn', 'lr', 'mlp', 'nbc', 'rf', 'svm')
    :param parameters: the model parameter space {'**parameter**': [**values**]}
    :param subset: a list of labels of features to be used
    :param is_test: a flag indicating whether the test or validation set is used
    :return:
    """

    path, directory = get_metrics_path(
        window_ms, stride_ms,
        imp_split, dos_type,
        model, parameters == {},
        subset, is_test=is_test)

    if not os.path.exists(directory):
        os.makedirs(directory)

    labels = ['Precision', 'Recall', 'TNR', 'FPR', 'FNR', 'Balanced accuracy', 'F1-score']

    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(['Class'] + labels)

        for key in metrics.keys():
            writer.writerow([key] + list(iter(metrics[key])))


def save_time(time_model, time_feature, window_ms, stride_ms,
              imp_split, dos_type, model, parameters, subset, is_test=False):
    """Saves specified time scores to a file with the name associated with the remaining parameters.

    :param time_model:
    :param time_feature:
    :param window_ms: the window size (int ms)
    :param stride_ms: the stride size (int ms)
    :param imp_split: the impersonation type (True, False)
    :param dos_type: the DoS type ('modified', 'original')
    :param model: the model name ('bn', 'dt', 'knn', 'lr', 'mlp', 'nbc', 'rf', 'svm')
    :param parameters: the model parameter space {'**parameter**': [**values**]}
    :param subset: a list of labels of features to be used
    :param is_test: a flag indicating whether the test or validation set is used
    :return:
    """

    path, directory = get_metrics_path(
        window_ms, stride_ms,
        imp_split, dos_type,
        model, parameters == {},
        subset, True, is_test)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(['Model_time', 'Feature_time', 'Total'])
        writer.writerow([time_model, time_feature, time_model + time_feature])