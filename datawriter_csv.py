import csv
import os
from metrics import get_metrics_path


def save_feature_durations(feature_durations, path, dir):
    """Saves specified feature duration dictionary to file at the specified path.
    Parameters are:
        - feature_durations: a dictionary connecting feature names to durations {'**feature**': **duration int ns**}
        - path:              the full file path
        - dir:               the directory path"""

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(['Feature', 'Time'])

        for feature in feature_durations.keys():
            writer.writerow([feature, feature_durations[feature]])


def save_metrics(metrics, period_ms, stride_ms, imp_split, dos_type, model, parameters, subset):
    """Saves specified metrics to a file with the name associated with the remaining parameters.
    Parameters are:
        - metrics:    a dictionary of scores {'**class**': (precision, recall, tnr, fpr, fnr, balanced_accuracy, f1)}
        - period_ms:  the window size (int ms)
        - stride_ms:  the stride size (int ms)
        - imp_split:  the impersonation type (True, False)
        - dos_type:   the DoS type ('modified', 'original')
        - model:      the model name ('bn', 'dt', 'knn', 'lr', 'mlp', 'nbc', 'rf', 'svm')
        - parameters: the model parameter space {'**parameter**': [**values**]}
        - subset:     a list of labels of features to be used"""

    path, dir = get_metrics_path(period_ms, stride_ms, imp_split, dos_type, model, parameters, subset)
    labels = ['Precision', 'Recall', 'TNR', 'FPR', 'FNR', 'Balanced accuracy', 'F1-score']

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(['Class'] + labels)

        for key in metrics.keys():
            writer.writerow([key] + list(iter(metrics[key])))


def save_time(time_model, time_feature, period_ms, overlap_ms, imp_split, dos_type, model, parameters, subset):
    path, dir = get_metrics_path(period_ms, overlap_ms, imp_split, dos_type, model, parameters, subset, True)

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(['Model_time', 'Feature_time', 'Total'])
        writer.writerow([time_model, time_feature, time_model + time_feature])