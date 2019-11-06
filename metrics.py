import datapoint
from operator import add


class Metrics():
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


class Result():
    period_ms: int
    stride_ms: int
    model: str
    imp_split: bool
    dos_type: str
    baseline: bool
    subset: str
    is_test: bool
    metrics: {}
    times: {}

    def __init__(self, period_ms, stride_ms, model, imp_split, dos_type, baseline, subset, is_test, metrics, times):
        self.period_ms = period_ms
        self.stride_ms = stride_ms
        self.model = model
        self.imp_split = imp_split
        self.dos_type = dos_type
        self.baseline = baseline
        self.subset = subset
        self.is_test = is_test
        self.metrics = metrics
        self.times = times


def filter_results(results, periods=None, strides=None, models=None,
                   imp_splits=None, dos_types=None, parameter_types=None, subsets=None):
    kept_results = []

    for result in results:
        if (periods is None or result.period_ms in periods) and \
                (strides is None or result.stride_ms in strides) and \
                (models is None or result.model in models) and \
                (imp_splits is None or result.imp_split in imp_splits) and \
                (dos_types is None or result.dos_type in dos_types) and \
                (parameter_types is None or result.baseline in parameter_types) and \
                (subsets is None or result.subset in subsets):

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


def get_metrics(y_test, y_predict):
    """
    Calculates the different metrics for the given given test set compared with the given predict set.

    :param y_test: A list of actual class labels
    :param y_predict: A list of predicted class labels
    :return: returns A dictionary of metrics of the form:
        - {'**class**': (precision, recall, tnr, fpr, fnr, balanced_accuracy, f1)}
    """

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
        metrics[key] = Metrics(*__get_metrics_tuple(counts[0], counts[1], counts[2], counts[3]))

    metrics['total'] = Metrics(*__get_macro_tuple(metrics))

    return metrics


def get_metrics_path(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset, is_time=False, is_test=False):
    """
    Finds the file path corresponding to the given parameters.

    :param period_ms: Window size (int ms)
    :param stride_ms: Stride size (int ms)
    :param imp_split: The impersonation type (True, False)
    :param dos_type: The DoS type ('modified', 'original')
    :param model: Model name ('bn', 'dt', 'knn', 'lr', 'mlp', 'nbc', 'rf', 'svm')
    :param baseline: Model parameter space {'**parameter**': [**values**]}
    :param subset: A list of labels of features to be used
    :param is_time: Whether the path is related to scores or durations (True, False)
    :param is_test: A flag indicating whether the test or validation set is used
    :return: returns the file path and directory path associated with the specified parameters.
    """

    imp_name = "imp_split" if imp_split else "imp_full"
    baseline_name = "baseline" if baseline else "selected_parameters"
    result_type = "test" if is_test else "validation"
    metric_type = "score" if is_time is False else "time"
    name = f"mixed_{result_type}_{metric_type}_{period_ms}ms_{stride_ms}ms"

    labels = list(datapoint.datapoint_attributes)[2:]

    for label in subset:
        name += f"_{labels.index(label)}"

    dir = f"result/{baseline_name}/{model}/{imp_name}/{dos_type}/"

    return dir + name + ".csv", dir


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


def __get_macro_tuple(metrics):
    # Finds the macro average of class scores in specified metric dictionary

    micros = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for label in metrics.keys():
        micros = list(map(add, micros, metrics[label]))

    length = len(metrics.keys())

    return [metric / length for metric in micros]