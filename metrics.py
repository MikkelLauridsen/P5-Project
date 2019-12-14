import math
import datapoint
from operator import add

import datareader_csv


class Metrics:
    precision: float
    recall: float
    tnr: float
    fpr: float
    fnr: float
    balanced_accuracy: float
    f1: float
    __iter_count: int

    def __init__(self, precision, recall, tnr, fpr, fnr, balanced_accuracy, f1):
        self.precision = precision
        self.recall = recall
        self.tnr = tnr
        self.fpr = fpr
        self.fnr = fnr
        self.balanced_accuracy = balanced_accuracy
        self.f1 = f1

    def __iter__(self):
        self.__iter_count = 0

        return self

    def __next__(self):
        if self.__iter_count > 6:
            raise StopIteration

        metric = [
            self.precision,
            self.recall,
            self.tnr,
            self.fpr,
            self.fnr,
            self.balanced_accuracy,
            self.f1][self.__iter_count]

        self.__iter_count += 1

        return metric


class Result:
    window_ms: int
    stride_ms: int
    model: str
    imp_split: bool
    dos_type: str
    baseline: bool
    subset: []
    is_test: bool
    metrics: {}
    times: {}

    def __init__(self, window_ms, stride_ms, model, imp_split, dos_type, baseline, subset, is_test, metrics, times):
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.model = model
        self.imp_split = imp_split
        self.dos_type = dos_type
        self.baseline = baseline
        self.subset = subset
        self.is_test = is_test
        self.metrics = metrics
        self.times = times


def filter_results(results, windows=None, strides=None, models=None, imp_splits=None,
                   dos_types=None, parameter_types=None, subsets=None, features=None,
                   without_features=None, f1_threshold=None, is_test=False):
    """Creates a list of filtered results from the specified list.

    :param results: a list of Result objects.
    :param windows: a list of window sizes (int ms).
    :param strides: a list of step-sizes (int ms).
    :param models: a list of model labels ('bn', 'dt', 'knn', 'lr', 'mlp', 'nbc', 'rf', 'svm').
    :param imp_splits: a list of flags indicating whether the dataset used had split impersonation labels.
    :param dos_types: a list of DoS type labels ('modified', 'original').
    :param parameter_types: a list of flags indicating whether the model baseline was used.
    :param subsets: a list of feature label lists, indicating the acceptable subsets used [[**labels**],..].
    :param features: a list of feature labels, designating features which must be in all subsets.
    :param without_features: a list of feature labels, designating features which may not be in the subsets.
    :param f1_threshold: a tuple of a metric type and a least F1-value (float).
    :param is_test: a flag indicating whether the results are based on the validation or test sets
    :return: returns a new list containing the filtered results.
    """

    kept_results = []

    for result in results:
        if (windows is None or result.window_ms in windows) and \
                (strides is None or result.stride_ms in strides) and \
                (models is None or result.model in models) and \
                (imp_splits is None or result.imp_split in imp_splits) and \
                (dos_types is None or result.dos_type in dos_types) and \
                (parameter_types is None or result.baseline in parameter_types) and \
                (subsets is None or result.subset in subsets) and \
                (f1_threshold is None or result.metrics[f1_threshold[0]].f1 >= f1_threshold[1]) and \
                (result.is_test == is_test):

            discard = False

            if features is not None:
                for label in features:
                    if label not in result.subset:
                        discard = True

                        break

            if without_features is not None:
                for label in without_features:
                    if label in result.subset:
                        discard = True

                        break

            if not discard:
                kept_results.append(result)

    return kept_results


def print_metrics(metrics):
    """Outputs a classification report to console, based on specified metrics"""

    labels = ['Precision', 'Recall', 'TNR', 'FPR', 'FNR', 'Balanced accuracy', 'F1-score']
    print("{:15}".format(" ") + "".join(["{:>18}".format(f"{label} ") for label in labels]))

    for key in metrics.keys():
        line = "{:>15}".format(f"{key}  ")

        for metric in iter(metrics[key]):
            line += "{:17.4f}".format(metric) + " "

        print(line)


def get_error_metrics():
    """Returns a dictionary of metrics of the form
    {'**class**': (precision, recall, tnr, fpr, fnr, balanced_accuracy, f1)},
    where each metric has score 0.0."""

    metrics = {}
    keys = ['normal', 'dos', 'fuzzy', 'impersonation', 'weighted', 'macro']

    for key in keys:
        metrics[key] = Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return metrics


def get_metrics(y_actual, y_predict):
    """Returns a dictionary of metrics of the form
    {'**class**': (precision, recall, tnr, fpr, fnr, balanced_accuracy, f1)}.

    :param y_actual: a list of actual class labels.
    :param y_predict: a list of predicted class labels.
    :return: a dictionary of Metrics objects.
    """

    if len(y_actual) != len(y_predict):
        raise IndexError()

    # Setup initial counters of true positives, false positives, true negatives, false negatives
    class_counters = {
        'normal': [0, 0, 0, 0],
        'dos': [0, 0, 0, 0],
        'fuzzy': [0, 0, 0, 0],
        'impersonation': [0, 0, 0, 0]}

    sample_sizes = {'normal': 0, 'dos': 0, 'fuzzy': 0, 'impersonation': 0}

    # For each pair of (actual label, predicted label), increment counters based on difference/equality
    for i in range(len(y_actual)):
        sample_sizes[y_actual[i]] += 1

        if y_actual[i] == y_predict[i]:
            __increment_equal(y_actual[i], class_counters)
        else:
            __increment_not_equal(y_actual[i], y_predict[i], class_counters)

    metrics = {}

    # For each class, use corresponding counters to calculate scoring metrics
    for key in class_counters.keys():
        counts = class_counters[key]
        metrics[key] = Metrics(*__get_metrics_tuple(counts[0], counts[1], counts[2], counts[3]))

    # Calculate overall score as a weighted average of the
    weighted = Metrics(*__get_weighted_metrics_tuple(metrics, sample_sizes))
    macro = Metrics(*__get_macro_metrics_tuple(metrics))
    metrics['weighted'] = weighted
    metrics['macro'] = macro

    return metrics


def load_result(window_ms, stride_ms, imp_split, dos_type, model, baseline, subset, is_test):
    metrics = datareader_csv.load_metrics(window_ms, stride_ms, imp_split, dos_type, model, baseline, subset, is_test)
    times = datareader_csv.load_times(window_ms, stride_ms, imp_split, dos_type, model, baseline, subset, is_test)

    return Result(window_ms, stride_ms, model, imp_split, dos_type, baseline, subset, is_test, metrics, times)


def get_result_path(result, is_test):
    return get_metrics_path(
        result.window_ms, result.stride_ms,
        result.imp_split, result.dos_type,
        result.model, result.baseline,
        result.subset, False, is_test)


def get_metrics_path(window_ms, stride_ms, imp_split, dos_type, model, baseline, subset, is_time=False, is_test=False):
    """Returns the file path and directory path associated with the specified parameters.

    :param window_ms: window size (int ms).
    :param stride_ms: stride size (int ms).
    :param imp_split: the impersonation type (True, False).
    :param dos_type: the DoS type ('modified', 'original').
    :param model: model name ('bn', 'dt', 'knn', 'lr', 'mlp', 'nbc', 'rf', 'svm').
    :param baseline: a flag indicating whether baseline parameters were used.
    :param subset: a list of labels of features that were used.
    :param is_time: a flag indicating whether the path is related to scores or durations (True, False).
    :param is_test: a flag indicating whether the test or validation set is used.
    :return: the file and directory paths.
    """

    imp_name = "imp_split" if imp_split else "imp_full"
    baseline_name = "baseline" if baseline else "selected_parameters"
    result_type = "test" if is_test else "validation"
    metric_type = "score" if is_time is False else "time"
    name = f"mixed_{result_type}_{metric_type}_{window_ms}ms_{stride_ms}ms"

    labels = datapoint.datapoint_features.copy()

    for label in subset:
        name += f"_{labels.index(label)}"

    dir = f"result/{baseline_name}/{model}/{imp_name}/{dos_type}/"

    return dir + name + ".csv", dir


def get_result_feature_breakdown(result: Result, type='validation'):
    """
    Gets a dict mapping feature labels to feature times. This is loaded according to a specific Result
    :param result: The result to load feature times to
    :param type: The type of feature times to load. Valid values are 'validation' and 'test'
    :return:
    """
    imp_name = "imp_split" if result.imp_split else "imp_full"
    name = f"mixed_{type}_time_{result.window_ms}ms_{result.stride_ms}ms"
    directory = f"data/feature/{imp_name}/{result.dos_type}/"

    path = directory + name + ".csv"
    return datareader_csv.load_feature_durations(path)


def __increment_equal(label, class_counters):
    # Updates specified class counters dictionary based on specified label
    class_counters[label][0] += 1

    for key in class_counters.keys():
        if key != label:
            class_counters[key][2] += 1


def __increment_not_equal(label, prediction, class_counters):
    # Updates specified class counters dictionary based on actual and predicted labels, which are different
    class_counters[label][3] += 1
    class_counters[prediction][1] += 1

    for key in class_counters.keys():
        if key != label and key != prediction:
            class_counters[key][2] += 1


def __get_metrics_tuple(tp, fp, tn, fn):
    # Returns a list of evaluation metrics based on specified classification counters
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1 if tp + fn == 0 else tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    balanced_accuracy = (recall + tnr) / 2
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return precision, recall, tnr, fpr, fnr, balanced_accuracy, f1


def __get_weighted_metrics_tuple(metrics, sample_sizes):
    # Finds the weighted average of class scores in specified metric dictionary
    micros = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # For each class, add weighted metrics to the list of summed metrics (micros)
    for label in metrics.keys():
        micros = list(map(add, micros, [sample_sizes[label] * metric for metric in metrics[label]]))

    length = math.fsum(sample_sizes.values())

    return [metric / length for metric in micros]


def __get_macro_metrics_tuple(metrics):
    # Finds the macro average of class scores in specified metric dictionary
    micros = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # For each class, add weighted metrics to the list of summed metrics (micros)
    for label in metrics.keys():
        micros = list(map(add, micros, metrics[label]))

    num_classes = len(metrics.keys())

    return [metric / num_classes for metric in micros]